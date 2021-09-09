class JoinFunction(object):
    """Customized join function must have an unique name attribute and a method named 
       compute_distance as below. The constructor can be overwritten"""
    def __init__(self):
        self.name = "jf_example"
        pass

    def compute_distance(self, left, right, LL_blocked, LR_blocked,
                         cache_dir=None):
        """Compute the distance of each tuple pair in the LL and LR blocked table.

        Parameters
        ----------
        left: pd.DataFrame
            A subset of the left table that contains the id column and the
            column to be processed. The id column is named as autofj_id.
            The column to be processed is named as value.

        right: pd.DataFrame
            A subset of the right table that contains the id column and the
            column to be processed. The id column is named as autofj_id.
            The column to be processed is named as value.

        LL_blocked: pd.DataFrame
            The LL blocked table that consists of the id columns and
            the columns to be processed. The id columns are named as
            autofj_id_l and autofj_id_r. The column to be processed is named as
            value_l and value_r.

        LR_blocked: pd.DataFrame
            The LR blocked table that consists of the id columns and
            the columns to be processed. The id columns are named as
            autofj_id_l and autofj_id_r. The column to be processed is named as
            value_l and value_r.

        Returns
        -------
        LL_distance: pd.Series
            Distance of each tuple pair in the LL blocked table.

        LR_distance: pd.Series
            Distance of each tuple pair in the LR blocked table.
        """
        LL_distance = None
        LR_distance = None
        return LL_distance, LR_distance
