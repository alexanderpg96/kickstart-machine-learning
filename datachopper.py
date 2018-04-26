from datetime import date

f = open("ks-projects-201612.csv","r")
f2 = open("Choppeddatasetv4.csv","w+")
f3 = open ("labels.csv","w+")
d=0
limit = 61725 #magic number for exactly 50000 samples. 11725 get filtered out.
listofcategories = []
for line in f:
    d+=1
    if d == 1 : continue #skip first line that has data labels
    splitline = line.split(",") #tokenize line
    if len(splitline) > 17: continue #filter out improper splits
    if splitline[13] is not "": continue #filter out improper splits
    if d>limit: break
    if "canc" in splitline[9]: continue #filter out canceled
    if "susp" in splitline[9]: continue  #filter out live
    if "live" in splitline[9]: continue  #filter out suspended
    listofcategories.append(splitline[2])
setofcategories = sorted(set(listofcategories))

d=0

for line in f:
    d+=1
    if d == 1 : continue #skip first line that has data labels
    splitline = line.split(",") #tokenize line
    if len(splitline) > 17: continue #filter out improper splits
    if splitline[13] is not "": continue #filter out improper splits
    if d>limit: break
    if "canc" in splitline[9]: continue #filter out canceled
    if "susp" in splitline[9]: continue  #filter out live
    if "live" in splitline[9]: continue  #filter out suspended
    if "fail" in splitline[9]:
        f3.write("0\n")
    if "succ" in splitline[9]:
        f3.write("1\n")
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
        splitline[2] = str(setofcategories.index(splitline[2]))
        rejoinedline = ",".join(splitline)
        f2.write(rejoinedline)
    except:
        continue