import tkinter as tk
from Gui import PenguinsGUI

def main():
    root = tk.Tk()
    app = PenguinsGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()