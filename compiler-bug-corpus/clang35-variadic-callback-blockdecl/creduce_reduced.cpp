#include <string>
template < typename Lambda > auto make_callback(Lambda l) {
  static auto p = l;
  return [](auto... args) { (p); };
}
class Button {
public:
  Button(std::string ) {}
  void wire() {
    void (*__trans_tmp_1)(int) = make_callback([](int ) {});
  }

std::string label_;
};
int main() {
  Button b("ok");
  b.wire();
}
