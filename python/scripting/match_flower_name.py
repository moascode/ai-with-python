# Write your code here

# HINT: create a dictionary from flowers.txt
def create_flowerdict(filename):
    flowers = {}
    with open(filename, 'r') as flowers_data:
        for flower in flowers_data:
            letter_to_flower = flower.strip().split(': ')
            flowers[letter_to_flower[0].lower()] = letter_to_flower[1]
    return flowers

# HINT: create a function to ask for user's first and last name
def ask_user():
    name = input('Enter your First [space] Last name only: ')
    return name[0].lower()
    
def main():
    flowers = create_flowerdict('flowers.txt')
    name = ask_user()
    # print the desired output
    print('Unique flower name with the first letter: {}'.format(flowers.get(name)))

main()