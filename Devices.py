

from tkinter import image_names


class Device():
    def __init__(self) -> None:
        pass


    def lanzar(self):
        ejecuta todo lo del colab
        guarda el modelo en el path que se le pasa en el constructor
        y guarda su accuracy

    def evaluate(self, path):
        ejecuta el evaluate que hay en colab con sus datos
        para ello crea otra vez sus temporales de imagenes y se evalua con esas image_names
        el accuracy del nuevo modelo se compara con el anterior y finalmente descarta el modleo con menos accuracy y nos dice con cual se queda