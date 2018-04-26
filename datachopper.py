from datetime import date
#
# d0 = date(2008, 8, 18)
# d1 = date(2008, 9, 26)
# delta = d1 - d0
# print delta.days

f = open("ks-projects-201612.csv","r")
f2 = open("Choppeddatasetv3.csv","w+")
d=0
for line in f:
    d+=1
    if d == 1 : continue #skip first line that has data labels
    splitline = line.split(",") #tokenize line
    if "canc" in splitline[9]: continue #filter out canceled
    try:
        deadline = splitline[5]
        launched = splitline[7]
        deadline = deadline.split(" ")[0].split("-")
        launched = launched.split(" ")[0].split("-")
        deadlinedate = date(int(deadline[0]), int(deadline[1]), int(deadline[2]))
        launcheddate = date(int(launched[0]), int(launched[1]), int(launched[2]))
        numdays = deadlinedate-launcheddate
        numdays = numdays.days #number of days in between both dates as an int
        splitline[15] = str(numdays)
        rejoinedline = ",".join(splitline)
        f2.write(rejoinedline)
    except:
        continue
    if d>60000: break