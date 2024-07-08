# like lesson before mostly a class use for organizing a funtion or in class called as methode when the function are complex

import random

class Hat:

    houses = ["bogor", "jakarta", "cibubur"] # this line is call "class variable"

    @classmethod
    def sort(cls, name): #cause we dont have __init__ and only has classmethod so it change from self to cls(basicly class)
        print(name, "is in" , random.choice(cls.houses))


Hat.sort("Alif")