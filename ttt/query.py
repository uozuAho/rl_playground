""" query trained models """

import typing as t
import json
import duckdb
import numpy as np

FILE = 'trained_models/tmcts_sym_100k_30'

duckdb.sql('create table d (board text, value float)')

PRINT_2D = True   # print boards in 2d
# PRINT_2D = False

def main():
    load_data()

    # print_bv_query("""
    # select * from d
    # where value < 1.0
    # order by value desc
    # limit 10;
    # """)

    # print_symmetrics("o.xxo..ox")

    table_quality_report()

    # gv = group_all_symmetry_values(FILE)
    # i = 0
    # for b, vs in gv.items():
    #     i += 1
    #     if i > 10: break
    #     vs = sorted(vs)
    #     avg = sum(vs)/len(vs)
    #     maxdiff = abs(vs[0] - vs[-1])
    #     print(b, [f'{x:6.3f}' for x in vs], f'avg: {avg:6.3f}  maxdiff: {maxdiff:6.3f}')


def table_quality_report():
    """ Attempt to succinctly show how good/bad the table is """
    print("Num boards: ")
    print(duckdb.sql("""select count(*) from d""").fetchone()[0])

    print("Num winning boards: ")
    print(duckdb.sql("""
    select count(*) from d
    where value = 1.0
    """).fetchone()[0])

    top10 = [r[0] for r in duckdb.sql("""
    select board from d
    order by value desc
    limit 10;
    """).fetchall()]

    print("Average num symmetrics for 10 winning boards (should be 8):")
    print(sum([len(load_symmetrics(b)) for b in top10])/10)

    top_nearly_wins = [r[0] for r in duckdb.sql("""
    select board from d
    where value < 1.0
    order by value desc
    limit 10;
    """).fetchall()]

    print("Average num symmetrics for top 10 nearly winning boards (should be 8):")
    print(sum([len(load_symmetrics(b)) for b in top_nearly_wins])/10)

    print("Symmetric values for top 10 nearly wins (should all be close to 1.0)")
    for b in top_nearly_wins:
        print([f'{v:0.2f}' for _,v in load_symmetrics(b)])

    print("any missing symmetries in table for top 10 nearly wins:")
    for b in top_nearly_wins:
        db_syms = set(s[0] for s in load_symmetrics(b))
        missing = set(symmetrics(b)) - db_syms
        if len(missing) > 0:
            print(missing)
        else:
            print("no")


def group_all_symmetry_values(path):
    """ for all boards in a table, return:
        {board: [values of all symmetries]}
    """
    with open(path) as infile:
        data = json.load(infile)

    asdf = {}
    for b,v in data.items():
        sym_values = [v]
        for symb in symmetrics(b):
            if symb in data:
                sym_values.append(data[symb])
        asdf[b] = sym_values
    return asdf


def print_bv_query(query: str):
    print_bv_pairs(duckdb.sql(query).fetchall())


def print_symmetrics(board: str):
    results = load_symmetrics(board)
    print(len(results), "values")
    print_bv_pairs(results)


def load_symmetrics(board):
    quoted_list = ','.join(f"'{s}'" for s in symmetrics(board))
    result = duckdb.sql(f"""
    select * from d
    where board in ({quoted_list})
    order by value desc
    limit 10;
    """, )

    results = result.fetchall()
    return results


def print_bv_pairs(bvs: t.List[t.Tuple[str, float]]):
    for b,v in bvs:
        print(to2d(b) if PRINT_2D else b,v)
        print()


def load_data():
    with open(FILE) as infile:
        data = json.load(infile)

    duckdb.executemany("insert into d values (?, ?)",
                       [[b.replace('|', ''), v] for b,v in data.items()])


def to2d(board: str):
    assert len(board) == 9
    return f'{board[:3]}\n{board[3:6]}\n{board[6:]}'


def symmetrics(board: str):
    assert len(board) == 9
    nboard = np.array([ord(c) for c in board]).reshape((3,3))
    symmetries: list[np.ndarray] = []

    # Identity
    symmetries.append(nboard)

    # Rotations
    for k in range(1, 4):
        symmetries.append(np.rot90(nboard, k))

    # Reflections
    symmetries.append(np.fliplr(nboard))                # Horizontal reflection
    symmetries.append(np.flipud(nboard))                # Vertical reflection
    symmetries.append(np.transpose(nboard))             # Diagonal (main)
    symmetries.append(np.fliplr(np.transpose(nboard)))  # Diagonal (anti)

    for s in symmetries:
        yield ''.join(chr(i) for i in s.reshape(9,))


if __name__ == "__main__":
    main()
    # for s in symmetrics("xx.....oo"):
    #     print(to2d(s))
    #     print()
