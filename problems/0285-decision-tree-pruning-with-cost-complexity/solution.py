
def compute_pruning_alphas(tree: dict) -> list:
    alphas = []

    def dfs(node):
        # Leaf node
        if node["left"] is None and node["right"] is None:
            return node["errors"], 1

        # Calculate left subtree
        left_errors, left_leaves = dfs(node["left"])

        # Calculate right subtree
        right_errors, right_leaves = dfs(node["right"])

        # Total leaf errors and count
        total_errors = left_errors + right_errors
        total_leaves = left_leaves + right_leaves

        # Effective alpha
        alpha = (node["errors"] - total_errors) / (total_leaves - 1)

        alphas.append(alpha)

        return total_errors, total_leaves

    dfs(tree)

    return sorted(alphas)
