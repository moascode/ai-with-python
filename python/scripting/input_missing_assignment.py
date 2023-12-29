names = input("Enter names separated by commas: ").split(",") # get and process input for a list of names
assignments = input("Enter assignments separated by commas: ").split(",") # get and process input for a list of the number of assignments
grades =  input("Enter grades separated by commas: ").split(",") # get and process input for a list of grades

## message string to be used for each student
## HINT: use .format() with this string in your for loop
message = "Hi {},\n\nThis is a reminder that you have {} assignments left to \
submit before you can graduate. Your current grade is {} and can increase \
to {} if you submit all assignments before the due date.\n\n"

student_set = zip(names, assignments, grades)

## write a for loop that iterates through each set of names, assignments, and grades to print each student's message
for name, assignment, grade in student_set:
    new_grade = int(grade) + int(assignment) * 2
    print(message.format(name, assignment, grade, new_grade))