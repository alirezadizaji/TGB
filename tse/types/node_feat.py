class NodeFeatType:
    CONSTANT = "CONSTANT"
    RAND = "RAND"
    ONE_HOT = "ONE_HOT"
    NODE_ID = "NODE_ID"

    @staticmethod
    def list():
        return [
            NodeFeatType.CONSTANT, 
            NodeFeatType.RAND, 
            NodeFeatType.ONE_HOT, 
            NodeFeatType.NODE_ID
        ]