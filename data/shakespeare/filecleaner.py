files = ["dev", "test", "train"]

for f in files:

    file = open(f"style.{f}.0", "r")
    writefile = open(f"style.{f}.2", "w")
    LIMIT = 32 - 1

    lines = []
    for line in file:
        # if line.strip(" \n") != "" :
        #     writefile.write(line)
        if line.count(" ") <= LIMIT:
            writefile.write(line)
        


