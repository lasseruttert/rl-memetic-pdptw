from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

class BestKnownSolutions:
    """A utility class to store and retrieve best known solutions for PDPTW instances."""
    
    BEST_KNOWN = {
        # Format: "instance_name": (best_known_num_vehicles, best_known_distance)
        
        # * 100 Node Instances
        # LC1 Class
        "lc101": (10, 828.94),
        "lc102": (10, 828.94),
        "lc103": (9, 1035.35),
        "lc104": (9, 860.01),
        "lc105": (10, 828.94),
        "lc106": (10, 828.94),
        "lc107": (10, 828.94),
        "lc108": (10, 826.44),
        "lc109": (9, 1000.60),
        
        # LC2 Class
        "lc201": (3, 591.56),
        "lc202": (3, 591.56),
        "lc203": (3, 591.17),
        "lc204": (3, 590.60),
        "lc205": (3, 588.88),
        "lc206": (3, 588.49),
        "lc207": (3, 588.29),
        "lc208": (3, 588.32),

        # LR1 Class
        "lr101": (19, 1650.80),
        "lr102": (17, 1487.57),
        "lr103": (13, 1292.68),
        "lr104": (9, 1013.39),
        "lr105": (14, 1377.11),
        "lr106": (12, 1252.62),
        "lr107": (10, 1111.31),
        "lr108": (9, 968.97),
        "lr109": (11, 1208.96),
        "lr110": (10, 1159.35),
        "lr111": (10, 1108.90),
        "lr112": (9, 1003.77),
        
        # LR2 Class
        "lr201": (4, 1253.23),
        "lr202": (3, 1197.67),
        "lr203": (3, 949.40),
        "lr204": (2, 849.05),
        "lr205": (3, 1054.02),
        "lr206": (3, 931.63),
        "lr207": (2, 903.06),
        "lr208": (2, 734.85),
        "lr209": (3, 930.59),
        "lr210": (3, 964.22),
        "lr211": (2, 911.52),

        # LRC1 Class
        "lrc101": (14, 1708.80),
        "lrc102": (12, 1558.07),
        "lrc103": (11, 1258.74),
        "lrc104": (10, 1128.40),
        "lrc105": (13, 1637.62),
        "lrc106": (11, 1424.73),
        "lrc107": (11, 1230.14),
        "lrc108": (10, 1147.43),

        # LRC2 Class
        "lrc201": (4, 1406.94),
        "lrc202": (3, 1374.27),
        "lrc203": (3, 1089.07),
        "lrc204": (3, 818.66),
        "lrc205": (4, 1302.20),
        "lrc206": (3, 1159.03),
        "lrc207": (3, 1062.05),
        "lrc208": (3, 852.76),
        
    }

    
    @classmethod
    def get_best_known(cls, instance_name: str) -> float:
        """Retrieve the best known fitness for a given instance.

        Args:
            instance_name: The name of the problem instance.

        Returns:
            The best known fitness value.

        Raises:
            KeyError: If the instance name is not found.
        """
        if instance_name in cls.BEST_KNOWN:
            return cls.BEST_KNOWN[instance_name]
        else:
            raise KeyError(f"Best known solution for instance '{instance_name}' not found.")