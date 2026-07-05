"""A small inventory management module."""

from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Product:
    sku: str
    name: str
    price: float
    tags: list[str] = field(default_factory=list)


class Inventory:
    def __init__(self):
        self._stock: dict[str, int] = defaultdict(int)
        self._products: dict[str, Product] = {}

    def register(self, product: Product) -> None:
        self._products[product.sku] = product

    def receive(self, sku: str, quantity: int) -> None:
        if sku not in self._products:
            raise KeyError(f"unknown sku: {sku}")
        self._stock[sku] += quantity

    def ship(self, sku: str, quantity: int) -> None:
        if self._stock[sku] < quantity:
            raise ValueError(f"not enough stock for {sku}")
        self._stock[sku] -= quantity

    def total_value(self) -> float:
        return sum(
            self._products[sku].price * count for sku, count in self._stock.items()
        )

    def low_stock(self, threshold: int = 5) -> list[str]:
        return [sku for sku, count in self._stock.items() if count < threshold]


def demo() -> float:
    inv = Inventory()
    inv.register(Product("A1", "Widget", 2.5, tags=["hardware"]))
    inv.register(Product("B2", "Gadget", 9.99))
    inv.receive("A1", 100)
    inv.receive("B2", 3)
    inv.ship("A1", 10)
    return inv.total_value()


if __name__ == "__main__":
    print(demo())
