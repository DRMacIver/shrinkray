#include <algorithm>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

template <typename T>
class Graph {
public:
    void add_edge(const T &from, const T &to) {
        adjacency_[from].push_back(to);
        adjacency_[to].push_back(from);
    }

    std::vector<T> bfs(const T &start) const {
        std::vector<T> order;
        std::unordered_map<T, bool> seen;
        std::queue<T> pending;
        pending.push(start);
        seen[start] = true;
        while (!pending.empty()) {
            T current = pending.front();
            pending.pop();
            order.push_back(current);
            auto it = adjacency_.find(current);
            if (it == adjacency_.end()) {
                continue;
            }
            for (const T &neighbour : it->second) {
                if (!seen[neighbour]) {
                    seen[neighbour] = true;
                    pending.push(neighbour);
                }
            }
        }
        return order;
    }

private:
    std::unordered_map<T, std::vector<T>> adjacency_;
};

int main() {
    Graph<int> g;
    g.add_edge(1, 2);
    g.add_edge(1, 3);
    g.add_edge(2, 4);
    g.add_edge(3, 4);
    auto order = g.bfs(1);
    for (int node : order) {
        std::cout << node << ' ';
    }
    std::cout << '\n';
    return 0;
}
