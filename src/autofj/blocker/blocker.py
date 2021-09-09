class Blocker(object):
    """The customized blocker must have a block method as below. The constructor
    can be overwritten"""
    def __init__(self):
        pass

    def block(self, left_table, right_table, id_column):
        """ Perform blocking on two tables

        Parameters
        ----------
        left_table: pd.DataFrame
            Reference table. The left table is assumed to be almost
            duplicate-free, which means it has no or only few duplicates.

        right_table: pd.DataFrame
            Another input table.

        id_column: string
            The name of id column in two tables.

        Returns:
        --------
            result: pd.DataFrame
            A table of records pairs survived blocking. Column names
            id_column + "_l" and id_column + "_r"
        """
        result = None
        return result