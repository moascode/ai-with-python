new_file = open("dummy_file.txt", "w")
new_file.write("Hi! \nThis is a dummy file")
new_file.close()

with open("dummy_file.txt", "a") as modify_file:
    modify_file.write("\nThis is appended line")

with open("dummy_file.txt", "r") as read_file:
    print(read_file.readline())
    print(read_file.read(3))
    print(read_file.readline())
    print(read_file.read())