// Reduced from a serialization library. Triggers a gcc 4.9 ICE
// (create_tmp_var, at gimple-expr.c) when an auto-returning variadic
// member template returns a lambda capturing its parameter pack by
// copy, and is called from a constructor with a non-trivially-copyable
// argument. gcc PR77739.
#include <cstddef>
#include <string>
#include <vector>

namespace serial {

struct Blob {
  Blob();
  Blob(const Blob &);
};

template <typename T>
struct Field {
  T value;
  const char *name;
};

class Encoder {
public:
  Encoder();

  template <typename... Args> auto encode(Args &&... args) {
    return [=] { write(args...); };
  }

  void write(Blob, const char *label);

  std::size_t bytes_written() const { return written_; }

private:
  std::vector<char> buffer_;
  std::size_t written_ = 0;
};

Encoder::Encoder() { encode(Blob(), ""); }

void Encoder::write(Blob, const char *) {}

}  // namespace serial

int main() {
  serial::Encoder enc;
  return static_cast<int>(enc.bytes_written());
}
