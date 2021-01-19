places_list = ['Berlin', 'Cape Town', 'Sydney', 'Moscow']

with open('test.txt', 'w') as filehandle:
    filehandle.writelines("%s  \n" % place for place in places_list)