// Reduced from a generic graph-traversal container. Triggers a gcc 4.9
// ICE (segfault in the C++ front end) when a class template member
// creates a generic lambda that calls a sibling member, and passes
// that lambda to another member template. gcc PR64382.
#include <cstddef>
#include <vector>

namespace graph {

template <typename Node>
struct Visitor {
  int visited = 0;
  void mark(Node) { ++visited; }
};

template <typename T>
struct Queue {
  std::vector<T> storage;

  void push(T) {}

  void enqueue_visit() {
    auto L = [=](auto &&v) {
      push(v);
    };
    traverse(L);
  }

  template <typename F>
  void traverse(F &&f) {
    f(T());
  }

  std::size_t size() const { return storage.size(); }
};

}  // namespace graph

template struct graph::Queue<int>;

int main() {
  graph::Queue<int> q;
  q.enqueue_visit();
  return static_cast<int>(q.size());
}
