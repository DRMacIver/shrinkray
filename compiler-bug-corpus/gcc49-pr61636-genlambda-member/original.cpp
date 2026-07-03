// Reduced from a real-world event-dispatch layer. Triggers a gcc 4.9
// ICE (segfault in the C++ front end) when a generic lambda names a
// member function of the enclosing class. gcc PR61636.
#include <vector>
#include <string>
#include <utility>

namespace evt {

using Priority = int;

struct Handler {
  virtual ~Handler() {}
  virtual void handle(int code) = 0;
};

template <typename T>
struct Registry {
  std::vector<T> entries;
  void add(const T& t) { entries.push_back(t); }
  std::size_t size() const { return entries.size(); }
};

class Logger {
public:
  explicit Logger(std::string tag) : tag_(std::move(tag)) {}
  void warn(int) {}
  void info(const std::string&) {}
private:
  std::string tag_;
};

struct Base {
  void Bar(int);
};

class Dispatcher : Base {
public:
  Dispatcher() : logger_("dispatch"), count_(0) {}

  void Foo(int);
  using Base::Bar;
  template <typename T> void Baz(T);

  void run_all();

  int count() const { return count_; }

private:
  Logger logger_;
  Registry<int> registry_;
  int count_;
};

void Dispatcher::run_all() {
  auto lam = [&](auto asdf) { Foo(asdf); };
  lam(0);
  auto lam1 = [&](auto asdf) { Bar(asdf); };
  lam1(0);
  auto lam2 = [&](auto asdf) { Baz(asdf); };
  lam2(0);
  auto lam3 = [&](auto asdf) { Baz<int>(asdf); };
  lam3(0);
}

}  // namespace evt

int main() {
  evt::Dispatcher d;
  d.run_all();
  return d.count();
}
