from time import time, sleep



how_many_snakes = 1
snake_string = """
Welcome to Python3!

             ____
            / . .\\
            \  ---<
             \  /
   __________/ /
-=:___________/

<3, Juno
"""


#print(snake_string * how_many_snakes)


start_time = time()
sleep(30)
end_time = time()
print("start time: {}".format(start_time))
print("end_time: {}".format(end_time))
print("Elapsed time: {}".format(end_time - start_time))
