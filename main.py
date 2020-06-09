__author__ = "Sebastian Puchała, Jakub Kowalik"


import pandas as pd
from src.id3 import *


data = pd.read_csv("data/divorce.csv", sep=";")

features = data.drop("Class", axis=1).astype("category")
target = data.Class
X_train, X_test, Y_train, Y_test = simple_validation(features, target, 0.8)

def menu_id3():
    clear()
    model = DecisionTree_ID3()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    acc = test_accuracy(Y_test, predictions)
    depth = model.tree.depth()
    size = model.tree.size()
    print ("Utworzono drzewo id3:")
    model.tree.show()
    print ("Dokladnosc klasyfikacji na zbiorze testowym: " + str(acc) + "%")
    print ("Glebokosc drzewa: " + str(depth))
    print ("Rozmiar drzewa: " + str(size))
    key_input = input("Nacisnij dowolny klawisz, aby wrocic do menu glownego...")
    menu()


def menu_C4_5(prunning = True):
    clear()
    model = DecisionTree_C45(prunning)
    model.fit(X_train, Y_train)
    predictions = predict(X_test, model.tree)
    acc = test_accuracy(Y_test, predictions)
    depth = model.tree.depth()
    size = model.tree.size()
    print ("Utworzono drzewo C4.5:")
    model.tree.show()
    print ("Dokladnosc klasyfikacji na zbiorze testowym: " + str(acc) + "%")
    print ("Glebokosc drzewa: " + str(depth))
    print ("Rozmiar drzewa: " + str(size))
    key_input = input("Nacisnij dowolny klawisz, aby wrocic do menu glownego...")
    menu()



def tree_comparision():
    clear()
    print("ID3:")
    model1 = DecisionTree_ID3()
    model1.fit(X_train, Y_train)
    model1.tree.show()
    model2 = DecisionTree_C45(pruning=False)
    model2.fit(X_train, Y_train)
    print("C4.5 bez przycinania:")
    model2.tree.show()
    model3 = DecisionTree_C45(pruning=True)
    model3.fit(X_train, Y_train)
    print("C4.5 z przycinaniem:")
    model3.tree.show()
    key_input = input("Nacisnij dowolny klawisz, aby wrocic do menu glownego...")
    menu()

def menu():
    clear()
    key_input = ""
    print("||************************************************||")
    print("||===================== MENU =====================||\n")
    print("(1). Wyswietl dane \n")
    print("(2). Utworz drzewo\n")
    print("(3). Przetestuj algorytmy\n")
    print("(4). Porownanie drzew\n")
    print("(5). Utworz nowe zbiory uczace i testowe (wylosuj ponownie)\n")
    print("(q). Wyjdz z programu\n")
    print("||================================================||\n")
    key_input = input("Podaj opcje: ")
    while key_input!="q" and key_input!='1' and key_input!='2' and key_input!='3'and key_input!='4'and key_input!='5':
       key_input = input("Bledna opcja! Sproboj ponownie: ")
    if key_input == "1":
       print_data()
    if key_input == "2":
       createTree()
    if key_input == "3":
       testAlg()
    if key_input == "4":
       tree_comparision()
    if key_input == "5":
       new_data_split()
    if key_input == "q":
        print("Koniec programu. Dziekuje za skorzystanie")

def testAlg():
    clear()
    print("Dla kazdej z ponizszych opcji algorytm wykonuje 100 iteracji losujac dane uczace i testowe ze zbioru.")
    print("Nastepnie dla kazdego przypadku tworzone jest drzewo po czym")
    print("badana jest srednia dokladnosc, glebokosc oraz rozmiar dla wszystkich drzew")
    print("(1). Testuj algorytm ID3\n")
    print("(2). Testuj algorytm C4.5 (bez przycinania)\n")
    print("(3). Testuj algorytm C4.5 (z przycinanianiem)\n")
    print("(q). Wyjdz z programu\n")
    key_input = input("Podaj opcje: ")
    while key_input!="q" and key_input!='1' and key_input!='2' and key_input!='3':
        key_input = input("Bledna opcja! Sproboj ponownie: ") 
    if key_input == "1":
        clear()
        print("Testowanie w trakcie...")
        parameters = test_id3(100)
        clear()
        print ("Srednia dokladnosc klasyfikacji na zbiorze testowym: " + str(parameters["acc"]) + "%")
        print ("Srednia glebokosc drzewa: " +str(parameters["depth"]))
        print ("Sredni rozmiar drzewa: " +str(parameters["size"]))
        key_input = input("Nacisnij dowolny klawisz, aby wrocic do menu glownego...")
        menu()
    if key_input == "2":
        clear()
        print("Testowanie w trakcie...")
        parameters = test_C45(100, prunning = False)
        clear()
        print ("Srednia dokladnosc klasyfikacji na zbiorze testowym: " + str(parameters["acc"]) + "%")
        print ("Srednia glebokosc drzewa: " +str(parameters["depth"]))
        print ("Sredni rozmiar drzewa: " +str(parameters["size"]))
        key_input = input("Nacisnij dowolny klawisz, aby wrocic do menu glownego...")
        menu()
    if key_input == "3":
        clear()
        print("Testowanie w trakcie...")
        parameters = test_C45(100, prunning = True)
        clear()
        print ("Srednia dokladnosc klasyfikacji na zbiorze testowym: " + str(parameters["acc"]) + "%")
        print ("Srednia glebokosc drzewa: " +str(parameters["depth"]))
        print ("Sredni rozmiar drzewa: " +str(parameters["size"]))
        key_input = input("Nacisnij dowolny klawisz, aby wrocic do menu glownego...")
        menu()
    if key_input == "q":
        print("Koniec programu. Dziekuje za skorzystanie")

def createTree():
    clear()
    print("Z jakiego algorytmu skorzystac:")
    print("(1). Utworz drzewo za pomoca algorytmu ID3\n")
    print("(2). Utworz drzewo za pomoca algorytmu C4.5 (bez przycinania)\n")
    print("(3). Utworz drzewo za pomoca algorytmu C4.5 (z przycinanianiem)\n")
    print("(q). Wyjdz z programu\n")
    key_input = input("Podaj opcje: ")
    while key_input!="q" and key_input!='1' and key_input!='2' and key_input!='3':
        key_input = input("Bledna opcja! Sproboj ponownie: ") 
    if key_input == "1":
        menu_id3()
    if key_input == "2":
        menu_C4_5(prunning = False)
    if key_input == "3":
        menu_C4_5(prunning = True)
    if key_input == "q":
        print("Koniec programu. Dziekuje za skorzystanie")

def new_data_split():
    clear()
    global  X_train, X_test, Y_train, Y_test
    X_train, X_test, Y_train, Y_test = simple_validation(features, target, 0.8)
    print("Poprawnie utworzono zbiory.")
    key_input = input("Nacisnij dowolny klawisz, aby wrocic do menu glownego...")
    menu()

def test_id3(iter):
    acc = 0
    depth = 0
    size = 0
    for i in range(iter):
        X_train, X_test, Y_train, Y_test = simple_validation(features, target, 0.8)
        model = DecisionTree_ID3()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        acc += test_accuracy(Y_test, predictions)
        depth += model.tree.depth()
        size += model.tree.size()
    parameters = {
        "acc": acc / iter,
        "depth": depth / iter,
        "size": size / iter,
    }
    return parameters


def test_C45(iter, prunning=True):
    acc = 0
    depth = 0
    size = 0
    for i in range(iter):
        X_train, X_test, Y_train, Y_test = simple_validation(features, target, 0.8)
        model = DecisionTree_C45(prunning)
        model.fit(X_train, Y_train)
        predictions = predict(X_test, model.tree)
        acc += test_accuracy(Y_test, predictions)
        depth += model.tree.depth()
        size += model.tree.size()
    parameters = {
        "acc": acc / iter,
        "depth": depth / iter,
        "size": size / iter,
    }
    return parameters


def print_data():
    clear()
    print ("Zbior atrybutow uczacych:")
    print (X_train)
    print ("Zbior klas uczacych:")
    print (Y_train)
    print ("Zbior atrybutow testowych:")
    print (X_test)
    print ("Zbior klas testowych:")
    print (Y_test)
    key_input = input("Nacisnij dowolny klawisz, aby wrocic do menu glownego...")
    menu()



if __name__ == "__main__":
    menu()