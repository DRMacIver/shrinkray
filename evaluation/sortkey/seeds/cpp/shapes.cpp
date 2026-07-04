#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

namespace geometry {

struct Point {
    double x = 0.0;
    double y = 0.0;
};

class Shape {
public:
    virtual ~Shape() = default;
    virtual double area() const = 0;
};

class Circle : public Shape {
public:
    explicit Circle(double radius) : radius_(radius) {}
    double area() const override { return M_PI * radius_ * radius_; }

private:
    double radius_;
};

class Rectangle : public Shape {
public:
    Rectangle(double width, double height) : width_(width), height_(height) {}
    double area() const override { return width_ * height_; }

private:
    double width_;
    double height_;
};

double total_area(const std::vector<std::unique_ptr<Shape>> &shapes) {
    return std::accumulate(
        shapes.begin(), shapes.end(), 0.0,
        [](double acc, const std::unique_ptr<Shape> &shape) {
            return acc + shape->area();
        });
}

}  // namespace geometry

int main() {
    std::vector<std::unique_ptr<geometry::Shape>> shapes;
    shapes.push_back(std::make_unique<geometry::Circle>(1.0));
    shapes.push_back(std::make_unique<geometry::Rectangle>(2.0, 3.0));
    return static_cast<int>(geometry::total_area(shapes));
}
