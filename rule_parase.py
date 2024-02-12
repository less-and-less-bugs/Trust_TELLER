input_str = """conj_36 :- has_attr_2, has_attr_5, has_attr_9.
label(0) :- not conj_36.
conj_44 :- has_attr_4, has_attr_8, has_attr_9.
label(0) :- not conj_44.
conj_0 :- has_attr_6, has_attr_7, has_attr_8.
label(1) :- not conj_0."""

# map_rule = {"has_attr_1":"P_2", "has_attr_2":"P_3", "has_attr_3":"P_4", "has_attr_4":"P_5", "has_attr_5":"P_6",
#             "has_attr_6":"P_7", "has_attr_7":"P1",  "has_attr_8":"P1",  "has_attr_9":"P1",  "has_attr_10":"P8", "label(0)":"P_{true}",
#             "label(1)": "P_{false}", " :- " : " = ", ".":"", ", " : " and "}


map_rule = {"has_attr_1":"P_2", "has_attr_2":"P_3", "has_attr_3":"P_4", "has_attr_4":"P_5", "has_attr_5":"P_6",
            "has_attr_6":"P_1", "has_attr_7":"P1",  "has_attr_8":"P1",  "has_attr_9":"P8", "label(0)":"P_{true}",
            "label(1)": "P_{false}", " :- " : " = ", ".":"", ", " : " and "}


for key, value in map_rule.items():
    input_str = input_str.replace(key, value)


# print(input_str)
# Split the input into clauses
clauses = input_str.split("\n")

# Iterate over the clause
output_clause = []
true_clauses = []
false_clauses = []
for clause in clauses:
    if clause.startswith("P_{true}"):
        true_clauses.append(clause.replace("P_{true} = ", ""))
    elif clause.startswith("P_{false}"):
        false_clauses.append(clause.replace("P_{false} = ", ""))
    else:
        output_clause.append(clause)


    # Concatenate the clauses with "or"
true_clause_concatenated = "P_{true} = " + " or ".join(true_clauses)
false_clause_concatenated = "P_{false} = " + " or ".join(false_clauses)
output_clause.append(true_clause_concatenated )
output_clause.append(false_clause_concatenated)

for clause in output_clause:
    print(clause )