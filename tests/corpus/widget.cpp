// A fairly typical fragment of C++ that a compiler bug report might
// contain: namespaces, inheritance, constructors with initializer
// lists, templates, and a bunch of code that is irrelevant to the
// actual bug (here played by the marker "trigger_the_bug").
#include <cstddef>
#include <vector>

namespace app {
namespace detail {

template <typename T, typename Allocator = std::allocator<T>>
class Buffer {
public:
    Buffer(size_t capacity, T fill)
        : capacity_(capacity), data_(capacity, fill), refcount_(1) {
        for (size_t i = 0; i < capacity; i++) {
            data_[i] = fill;
        }
    }

    T &at(size_t index) { return data_[index]; }
    size_t capacity() const noexcept { return capacity_; }

private:
    size_t capacity_;
    std::vector<T, Allocator> data_;
    int refcount_;
};

} // namespace detail

class Shape {
public:
    virtual ~Shape() {}
    virtual double area() const = 0;
};

class Widget : public Shape, private detail::Buffer<double> {
public:
    Widget(double width, double height)
        : Buffer(16, 0.0), width_(width), height_(height) {}

    double area() const override { return width_ * height_; }

    void trigger_the_bug() { at(99) = compute_scale(width_, height_); }

private:
    static double compute_scale(double w, double h) { return w / h; }

    double width_;
    double height_;
};

} // namespace app

int main() {
    app::Widget w(3.0, 4.0);
    w.trigger_the_bug();
    return 0;
}
