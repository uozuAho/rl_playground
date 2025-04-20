""" query trained models """

import typing as t
import json
import duckdb
import numpy as np

FILE = 'trained_models/tabmcts_10k_20'

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

    symmetric_values("o.xxo..ox")


def print_bv_query(query: str):
    print_bv_pairs(duckdb.sql(query).fetchall())


def symmetric_values(board: str):
    quoted_list = ','.join(f"'{s}'" for s in symmetrics(board))
    result = duckdb.sql(f"""
    select * from d
    where board in ({quoted_list})
    order by value desc
    limit 10;
    """, )

    results = result.fetchall()
    print(len(results), "values")
    print_bv_pairs(results)


def print_bv_pairs(bvs: t.List[t.Tuple[str, float]]):
    for b,v in bvs:
        print(to2d(b) if PRINT_2D else b,v)
        print()


def load_data():
    with open(FILE) as infile:
        data = json.load(infile)

    duckdb.executemany(f"insert into d values (?, ?)",
                       [[b.replace('|', ''), v] for b,v in data.items()])


def to2d(board: str):
    assert len(board) == 9
    return f'{board[:3]}\n{board[3:6]}\n{board[6:]}'


def symmetrics(board: str):
    assert len(board) == 9
    nboard = np.array([ord(c) for c in board]).reshape((3,3))
    symmetries = []

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
