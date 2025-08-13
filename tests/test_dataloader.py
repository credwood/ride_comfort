from analysis.loaders import load_all_triax


def test_load_all_triax():
    # Assuming the directory 'data/' contains the necessary files for testing
    non_rave, rave = load_all_triax("data/", flip_x_y=False)
    assert isinstance(rave, dict)
    assert isinstance(non_rave, dict)
    assert len(rave) > 0
    assert len(non_rave) > 0

    for triax, dim_data in non_rave.items():
        if triax == "1" or triax == "2":
            continue
        for dim, data in dim_data.items():
            assert data["Description"].iloc[1].split(" ")[1] == triax
            letter_dim = data["Description"].iloc[2].split(" ")[0]
            if triax != "5":
                if letter_dim == "X":
                    assert dim == "1"
                elif letter_dim == "Y":
                    assert dim == "2"
                elif letter_dim == "Z":
                    assert dim == "3"
            else:
                if letter_dim == "X":
                    assert dim == "4"
                elif letter_dim == "Y":
                    assert dim == "5"
                elif letter_dim == "Z":
                    assert dim == "6"
    
    for triax, dim_data in rave.items():
        if triax == "1" or triax == "2":
            continue
        for dim, data in dim_data.items():
            assert data["Description"].iloc[1].split(" ")[1] == triax
            letter_dim = data["Description"].iloc[2].split(" ")[0]
            if triax != "5":
                if letter_dim == "X":
                    assert dim == "1"
                elif letter_dim == "Y":
                    assert dim == "2"
                elif letter_dim == "Z":
                    assert dim == "3"
            else:
                if letter_dim == "X":
                    assert dim == "4"
                elif letter_dim == "Y":
                    assert dim == "5"
                elif letter_dim == "Z":
                    assert dim == "6"

def test_load_all_triax_x_y_flip():
    # Assuming the directory 'data/' contains the necessary files for testing
    non_rave, rave = load_all_triax("data/", flip_x_y=True)
    assert isinstance(rave, dict)
    assert isinstance(non_rave, dict)
    assert len(rave) > 0
    assert len(non_rave) > 0

    for triax, dim_data in non_rave.items():
        for dim, data in dim_data.items():
            assert data["Description"].iloc[1].split(" ")[1] == triax
            letter_dim = data["Description"].iloc[2].split(" ")[0]
            if triax != "5":
                if letter_dim == "X":
                    assert dim == "2"
                elif letter_dim == "Y":
                    assert dim == "1"
                elif letter_dim == "Z":
                    assert dim == "3"
            else:
                if letter_dim == "X":
                    assert dim == "5"
                elif letter_dim == "Y":
                    assert dim == "4"
                elif letter_dim == "Z":
                    assert dim == "6"
    
    for triax, dim_data in rave.items():
        for dim, data in dim_data.items():
            assert data["Description"].iloc[1].split(" ")[1] == triax
            letter_dim = data["Description"].iloc[2].split(" ")[0]
            if triax != "5":
                if letter_dim == "X":
                    assert dim == "2"
                elif letter_dim == "Y":
                    assert dim == "1"
                elif letter_dim == "Z":
                    assert dim == "3"
            else:
                if letter_dim == "X":
                    assert dim == "5"
                elif letter_dim == "Y":
                    assert dim == "4"
                elif letter_dim == "Z":
                    assert dim == "6"