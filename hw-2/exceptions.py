"""
Объявите следующие исключения:
- LowFuelError
- NotEnoughFuel
- CargoOverload
"""
class LowFuelError(Exception):
    def __init__(self) -> None:
        super().__init__("Низкий уровень топлива")

class NotEnoughFuel(Exception):
    def __init__(self, current, required) -> None:
        super().__init__(f'Недостаточно топлива для поездки. Текущий уровень: {current} Требуется: {required}')

class CargoOverload(Exception):
    def __init__(self, cargo_weight, weight_allowed) -> None:
        super().__init__(f'Превышение допустимого веса груза. Вес груза: {cargo_weight} Допустимый вес: {weight_allowed}')
