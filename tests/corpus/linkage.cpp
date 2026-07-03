// Mixed linkage blocks and nested namespaces wrapping one interesting
// declaration (the marker is "poison_global").
namespace outer {
namespace inner {

extern "C" {
int poison_global = 42;

int helper_one(int x) { return x + 1; }
}

namespace {
int helper_two(int x) { return x + 2; }
}

} // namespace inner

namespace also::nested::deeply {
int helper_three(int x) { return x + 3; }
}

} // namespace outer
