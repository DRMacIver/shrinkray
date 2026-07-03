// Template-heavy C++ in the style of a frontend crash report. The
// marker is the "explode" method.
#include <type_traits>

template <typename T, int Rows, int Cols>
struct Matrix {
    T cells[Rows][Cols];

    template <typename U>
    Matrix<U, Rows, Cols> cast() const {
        Matrix<U, Rows, Cols> result;
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Cols; c++)
                result.cells[r][c] = static_cast<U>(cells[r][c]);
        return result;
    }

    T explode() const { return cells[Rows][Cols]; }
};

template <typename T>
using Square3 = Matrix<T, 3, 3>;

template <typename T, typename Enable = std::enable_if_t<std::is_integral<T>::value>>
T sum_all(const Matrix<T, 3, 3> &m) {
    T total = T();
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            total += m.cells[r][c];
    return total;
}

int main() {
    Square3<int> m = {};
    int unused = sum_all(m);
    (void)unused;
    return m.explode();
}
