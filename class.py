def main():
    student = Student.get()
#    if student["name"] == "alif":
#        student["house"]== "jakarta" # this line of code and 1 above this says: jika nama yang dilist disebutkan alif maka rumahnya automaticly jakarta and will overwrite if input incorrectly
#    print(f"{student['name']} is from {student['house']}")
    print(student)
#    print(f"{student.name} is from {student.house}")
#   print("Expecto patronum!")
#    print(student.charm())

# class function can act like a container that what's inside is a bunch of function that can be used
class Student:
    def __init__(self, name , house,) : # a dunder init function is always have to be at the start of making any classes then below it u can define whatever its needed
#        if not name:                         # from line 4 to 7 is basicly are backed on getter and setter function
#            raise ValueError("missing name")
#        if house not in ["jakarta","cibubur","bojong"]: # this 2 line of code basicly can be remove because
#            raise ValueError("invalid house") # its already define on setter house section
        self.name = name
        self.house = house
#        self.patronus = patronus

    def __str__(self) -> str:
        return (f"{self.name} is from {self.house}")
    
    @classmethod
    def get(cls):
        name = input("name :")
        house = input("house: ")
        return cls(name, house)
    ##    patronus = input("patronus")

    @property       # getter for name
    def name(self):
        return self._name

    @name.setter   # setter for name
    def name(self, name):
        if not name:
            raise ValueError("missing name")
        self._name = name # its returning "name" assignment to init method above

    @property       # getter for house
    def house(self):
        return self._house
    
    @house.setter   # setter
    def house(self, house):
        if house not in ["jakarta","cibubur","bojong"]:
            raise ValueError("invalid house")
        self._house = house # its returning "house" assignment to init method above

#in getter and setter method, the using of _ aka underscore are privately own to class parent 
# and dont call _ outside the parent class.

#    def charm(self):
#        match self.patronus:
#            case"Stag":
#                return"+-+"
#            case"otter":
#                return"+v+"
#            case _:
#                return"boo"

if __name__ == "__main__" :
    main()

