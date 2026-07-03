// Reduced from a signal/callback dispatch layer. Triggers a clang 3.5
// backend crash (CodeGenFunction::GetAddrOfBlockDecl) when a generic
// variadic lambda forwards its arguments through a pointer captured
// from an enclosing function template. LLVM/clang PR (c++/78816-style).
#include <vector>
#include <string>

namespace ui {

using Callback = void (*)(int);

struct Widget {
  std::string name;
  int id;
};

template <typename T>
struct Slot {
  T handler;
  bool enabled = true;
};

void install(Callback cb) {
  cb(42);
}

template <typename Lambda>
static auto make_callback(Lambda &&l) {
  static auto *p = &l;
  p = &l;
  return [](auto... args) { return (*p)(args...); };
}

class Button {
public:
  explicit Button(std::string label) : label_(std::move(label)) {}

  void wire() {
    int state = 5;
    install(make_callback([=](int y) { (void)state; (void)y; }));
  }

private:
  std::string label_;
};

}  // namespace ui

int main() {
  ui::Button b("ok");
  b.wire();
  return 0;
}
