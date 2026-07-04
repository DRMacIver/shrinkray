// Reduced from a compile-time units/format library that builds string
// literals into type-level character packs. Triggers a gcc 4.9 ICE
// (type_dependent_expression_p, at cp/pt.c) when a template default
// argument uses decltype of a character-pack user-defined literal.
#include <cstddef>

namespace units {

template <class C, C... S>
struct String_template {
  static constexpr std::size_t length = sizeof...(S);
};

template <class C, C... S>
constexpr String_template<C, S...> operator""_template() {
  return String_template<C, S...>{};
}

template <class T>
struct Quantity {
  T magnitude;
};

template <class prefix = decltype("0x"_template), class T>
int hex(T v) {
  return 1;
}

template <int v>
void render() {
  auto h = hex(2);
  (void)h;
}

}  // namespace units

int main() {
  units::render<0>();
  return 0;
}
