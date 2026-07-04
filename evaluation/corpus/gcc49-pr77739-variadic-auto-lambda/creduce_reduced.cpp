struct Blob {
  Blob();
  Blob(const Blob &);
};
class Encoder {
Encoder();
  template < typename... Args > auto encode(Args &&...args) {
    return [=] { write(args...); };
  }
  void write(Blob, const char *);
};
Encoder::Encoder() { encode(Blob(), ""); }
