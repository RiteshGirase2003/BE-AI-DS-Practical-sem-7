# Define basic fuzzy set operations

def fuzzy_union(A, B):
    """Union of two fuzzy sets"""
    return {x: max(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

def fuzzy_intersection(A, B):
    """Intersection of two fuzzy sets"""
    return {x: min(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

def fuzzy_complement(A):
    """Complement of a fuzzy set"""
    return {x: 1 - mu for x, mu in A.items()}

def fuzzy_difference(A, B):
    """Difference of two fuzzy sets (A - B)"""
    return {x: min(A.get(x, 0), 1 - B.get(x, 0)) for x in set(A) | set(B)}

def cartesian_product(A, B):
    """Cartesian product of two fuzzy sets to create a fuzzy relation"""
    return {(a, b): min(A[a], B[b]) for a in A for b in B}

def max_min_composition(R1, R2):
    """Max-min composition of two fuzzy relations"""
    composition = {}
    # Find unique first elements from R1 and second elements from R2
    X = set(a for a, _ in R1)
    Z = set(c for _, c in R2)
    
    for x in X:
        for z in Z:
            # Find all y such that (x, y) in R1 and (y, z) in R2
            candidates = []
            for (a, b1), val1 in R1.items():
                for (b2, c), val2 in R2.items():
                    if a == x and b1 == b2 and c == z:
                        candidates.append(min(val1, val2))
            if candidates:
                composition[(x, z)] = max(candidates)
            else:
                composition[(x, z)] = 0
    return composition

# Example fuzzy sets
A = {'a': 0.2, 'b': 0.7, 'c': 1.0}
B = {'a': 0.5, 'b': 0.4, 'd': 0.8}

# Perform operations
print("Fuzzy Set A:", A)
print("Fuzzy Set B:", B)

union = fuzzy_union(A, B)
print("\nUnion (A ∪ B):", union)

intersection = fuzzy_intersection(A, B)
print("\nIntersection (A ∩ B):", intersection)

complement_A = fuzzy_complement(A)
print("\nComplement (A'):", complement_A)

difference = fuzzy_difference(A, B)
print("\nDifference (A - B):", difference)

# Create fuzzy relations
relation_AB = cartesian_product(A, B)
relation_BA = cartesian_product(B, A)

print("\nFuzzy Relation (A × B):", relation_AB)
print("\nFuzzy Relation (B × A):", relation_BA)

# Max-min composition of relations
composition = max_min_composition(relation_AB, relation_BA)
print("\nMax-Min Composition (A × B) ∘ (B × A):", composition)
