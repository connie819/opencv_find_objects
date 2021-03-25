my_list = ["Apple", "Oranges", "Grapes", "Bananas"]

file = open('file_name.txt', 'w')

# Writing a list to the file. 
file.writelines(line + '\n' for line in my_list)

file.close()

with open('path_data' + str(itt) + '.txt', 'w') as filehandle:
    for place in range(len(list_x)):
        filehandle.writelines("%s," % int(list_x[place]) + "%s," % int(list_y[place]) + "%s\n" % int(list_z[place]))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_zlim(-63,200)
    ax.set_xlim(-50*0.6,450*0.6)
    ax.set_ylim(-350*0.6,150*0.6)
    # 產生 3D 座標資料
    z_array = np.array(list_z)
    x_array = np.array(list_x)
    y_array = np.array(list_y)