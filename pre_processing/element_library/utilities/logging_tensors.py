# pre_processing\element_library\utilities\logging_tensors.py

import numpy as np

def format_tensor_for_logging(tensor: np.ndarray, gauss_point: int, xi: float, weight: float, n_columns: int = 6) -> str:
    """
    Format a tensor into a multi-column string for efficient logging, with a header.
    If the tensor is 1D, it is written vertically. If 2D, it is written in multi-column format.

    Args:
        tensor (np.ndarray): The tensor to format (1D or 2D).
        gauss_point (int): Gauss point number.
        xi (float): Natural coordinate (xi) of the Gauss point.
        weight (float): Weight of the Gauss point.
        n_columns (int): Number of columns per row in the log (for 2D tensors).

    Returns:
        str: Formatted tensor string with header.
    """
    formatted_lines = []

    # Add header aligned with the first column
    header = f"Gauss Point {gauss_point}: ξ = {xi:.6e}, Weight = {weight:.6e}"
    formatted_lines.append(header)

    # Handle 1D tensors (vectors)
    if tensor.ndim == 1:
        for value in tensor:
            formatted_lines.append(f"    {value:.6e}")  # Write each value on a new line
    # Handle 2D tensors (matrices)
    else:
        rows, cols = tensor.shape
        for i in range(rows):
            row_data = tensor[i]
            # Split row into chunks of `n_columns`
            for j in range(0, cols, n_columns):
                chunk = row_data[j:j + n_columns]
                formatted_lines.append("    " + " ".join(f"{x:.6e}" for x in chunk))  # Indent tensor data

    return "\n".join(formatted_lines)


def log_tensor(logger, tensor: np.ndarray, gauss_point: int, xi: float, weight: float, description: str, n_columns: int = 6) -> None:
    """
    Log a tensor with a description, Gauss point information, and formatted output.

    Args:
        logger: The logger object to use for logging.
        tensor (np.ndarray): The tensor to log.
        gauss_point (int): Gauss point number.
        xi (float): Natural coordinate (xi) of the Gauss point.
        weight (float): Weight of the Gauss point.
        description (str): Description of the tensor (e.g., "Shape Function First Derivatives").
        n_columns (int): Number of columns per row in the log.
    """
    formatted_tensor = format_tensor_for_logging(tensor, gauss_point, xi, weight, n_columns)
    logger.debug(f"{description}:\n{formatted_tensor}")