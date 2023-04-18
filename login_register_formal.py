import tkinter as tk
import tkinter.messagebox as messagebox
import bcrypt
import json

# create a new Tkinter window
root = tk.Tk()
root.title("Login Form")

# create labels for the username and password fields
username_label = tk.Label(root, text="Username:")
password_label = tk.Label(root, text="Password:")

# create entry fields for the username and password
username_entry = tk.Entry(root)
password_entry = tk.Entry(root, show="*")

# place the labels and entry fields on the window
username_label.grid(row=0, column=0)
username_entry.grid(row=0, column=1)
password_label.grid(row=1, column=0)
password_entry.grid(row=1, column=1)

# create a function to handle the login button
def login():
    username = username_entry.get()
    password = password_entry.get()
    # load the user data from the file
    with open("users.json", "r") as f:
        users = json.load(f)
    # check if the user exists
    if username in users:
        # check if the password is correct
        hashed_password = users[username]["password"]
        if bcrypt.checkpw(password.encode("utf-8"), hashed_password):
            # open a new window with a welcome message
            welcome_window = tk.Toplevel()
            welcome_window.title("Welcome")
            welcome_label = tk.Label(welcome_window, text=f"Welcome, {username}!")
            welcome_label.pack()
        else:
            # display an error message box
            messagebox.showerror("Login Failed", "Incorrect password.")
    else:
        # display an error message box
        messagebox.showerror("Login Failed", "User not found.")

# create a function to handle the registration button
def register():
    username = username_entry.get()
    password = password_entry.get()
    # load the user data from the file
    with open("users.json", "r") as f:
        users = json.load(f)
    # check if the user already exists
    if username in users:
        messagebox.showerror("Registration Failed", "User already exists.")
    else:
        # hash the password using bcrypt
        hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        # add the new user to the user data
        users[username] = {"password": hashed_password.decode("utf-8")}
        # save the updated user data to the file
        with open("users.json", "w") as f:
            json.dump(users, f)
        messagebox.showinfo("Registration Successful", "Registration successful!")

# create login and registration buttons
login_button = tk.Button(root, text="Login", command=login)
register_button = tk.Button(root, text="Register", command=register)
login_button.grid(row=2, column=0)
register_button.grid(row=2, column=1)

# start the Tkinter event loop
root.mainloop()
