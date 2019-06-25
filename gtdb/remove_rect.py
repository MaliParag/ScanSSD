
def check_collision(rectA,  rectB):

    # returns True if A is inside B

    #left, top, right, bottom
    #If any of the sides from A are outside of B
    if rectA[3] <= rectB[1]:
        return False

    if rectA[1] >= rectB[3]:
        return False

    if rectA[2] <= rectB[0]:
        return False

    if rectA[0] >= rectB[2]:
        return False

    #If none of the sides from A are outside B
    return True
