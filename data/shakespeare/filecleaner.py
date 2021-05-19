file = open("style.test.1", "r")
writefile = open("style.test.2", "w")

lines = []
for line in file:
    if line.strip(" \n") != "" :
        writefile.write(line)

