# Results: shrink ray vs c-reduce on the compiler-bug corpus

Each entry was reduced from `original.cpp` by both shrink ray (its
pure-Python C/C++ passes) and c-reduce 2.11.0 (with its bundled
`clang_delta`), against the **same** real-compiler oracle running in
Docker. Every reduced file was re-verified to still crash its compiler
with the original signature.

## Sizes

`nows` = size with all whitespace stripped, which is the fair basis for
comparison: shrink ray normalizes identifiers to single letters and
removes all inter-token whitespace, whereas c-reduce leaves readable,
`clang-format`-ed output. Raw bytes therefore flatter shrink ray for
reasons unrelated to how much structure each removed.

| Entry | original | shrink ray | c-reduce | shrink `nows` | c-reduce `nows` |
|-------|---------:|-----------:|---------:|--------------:|----------------:|
| gcc49-pr61636-genlambda-member          | 1395 |  79 | 109 |  72 |  79 |
| gcc49-pr64382-genlambda-template-member |  879 | 144 | 208 | 134 | 157 |
| gcc49-pr77739-variadic-auto-lambda      | 1004 | 166 | 258 | 160 | 206 |
| gcc49-udlit-char-pack-template          |  834 | 126 | 118 | 122 |  80 |
| clang35-variadic-callback-blockdecl     | 1005 | 201 | 336 | 188 | 267 |

**Important caveat on the numbers.** Under amd64 emulation c-reduce's
long tail is prohibitively slow, so c-reduce was given a 15-minute
budget per entry (it front-loads its high-value passes); it ran to
completion only on `pr61636` and `udlit`. The `pr64382`, `pr77739`, and
`clang` c-reduce figures are therefore *time-capped*, not
capability-limited, and understate c-reduce. The clang reduction was
also cut off on the shrink-ray side for the same reason. So the raw
size ranking ("shrink ray wins 4/5") is **not** a fair capability
comparison — the useful signal is *structural*: what constructs each
tool could and couldn't remove.

## What c-reduce removed that shrink ray couldn't

Two of the fully-converged comparisons expose concrete missing
capabilities in `src/shrinkray/passes/cpp.py`. Both are cases where
c-reduce's `clang_delta` **rewrites** code rather than only deleting it.

### 1. Type replacement / simplification (the biggest gap)

shrink ray only ever *deletes* structure; it never *substitutes* a
type for a simpler one. On `udlit` (both tools converged), c-reduce
changed `operator""`'s return type from the class template `a<A>` to
the plain char type and then deleted the whole `class a` — reaching 80
`nows` chars vs shrink ray's 122, which kept the forward-declared class
because the return type still referenced it:

```
shrink ray:  template<class>class a;template<class A,A...>a<A> operator""_template();
             template<class=decltype(0_template),class a>a A(a){A(0)
c-reduce:    template<class b,b...>b operator""c();
             template<class=decltype(""c)>void d(int){d(2)
```

The same pattern shows on `pr64382`: c-reduce replaced the class
template's type parameter `T` with `int` throughout (`push(int)`,
`f(int())`), which shrink ray cannot do.

clang_delta analogues: `empty-struct-to-int`, `template-arg-to-int`,
`replace-class-with-base-template-spec`, `union-to-struct`,
`reduce-pointer-level`. A "replace a user type with a builtin / with a
template argument" pass, tried speculatively like the other passes,
would close most of this gap.

### 2. Namespace-qualifier rewriting

`remove_namespaces` only splices the `namespace X { … }` braces. If any
`X::name` reference survives elsewhere — e.g. an explicit instantiation
`template struct X::f<int>;` — splicing dangles it, the candidate fails
to compile, and the namespace is kept. c-reduce's `remove-namespace`
also strips the `X::` qualifier from references. Clear on `pr64382`:

```
shrink ray:  namespace a{template<typename A>struct f{…};}template struct a::f<int>;
c-reduce:    template<typename>struct Queue{…};template struct Queue<int>;
```

The same limitation kept the `namespace a{…}` (and `a::n`) in the clang
reduction. Fix: when removing a namespace, also delete `X::` prefixes on
references to names that were declared in it.

### 3. Base-class collapse (minor)

On `pr61636` shrink ray kept a two-class base/derived structure
(`struct c{c b(int);};class a:c{…}`) because the lambda calls a member
that lives in the base; `remove_base_classes` can drop the base-class
list but can't relocate the base's members into the derived class. It
was still smaller here overall, so this is low priority.

## Takeaway

On these bugs shrink ray produces reductions that are structurally on
par with c-reduce and, thanks to identifier/whitespace normalization,
usually smaller in raw bytes. The comparison's real value is the two
actionable gaps above — **type replacement** and **namespace-qualifier
rewriting** — neither of which the current deletion-only passes can
express, and both of which c-reduce handles via `clang_delta`
rewrites. These are the highest-value additions to
`src/shrinkray/passes/cpp.py`.
