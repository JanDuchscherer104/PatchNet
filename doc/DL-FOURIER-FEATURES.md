# Learnable Fourier Features for Spatial Representations
### Parameters Explained

1. **`pos_dim (int)`**:
    - **Description**: This is the dimensionality of the input position. For example, in a 2D spatial grid, `pos_dim` would be 2, representing (x, y) coordinates.
    - **Optimal Value**: Typically, this is determined by the nature of the spatial data. For 2D images, `pos_dim = 2` is common. For 3D spatial data, `pos_dim = 3`, and so on.

2. **`f_dim (int)`**:
    - **Description**: This is the dimensionality of the Fourier features. Higher values allow the network to capture more detailed spatial relationships by embedding more frequency components.
    - **Optimal Value**: The paper suggests using higher `f_dim` values to capture high-frequency details effectively. Common values are 256, 512, or 768. A default of 768 is chosen for its balance between detail and computational cost.

3. **`h_dim (int)`**:
    - **Description**: This is the dimensionality of the hidden layer in the MLP. It determines the capacity of the MLP to process the Fourier features before outputting the final embedding.
    - **Optimal Value**: Values between 32 and 128 are commonly used. The default of 32 is chosen based on the balance between network complexity and performance as suggested in the paper.

4. **`d_dim (int)`**:
    - **Description**: This is the dimensionality of the output embedding. It should match the input dimensionality required by subsequent layers (e.g., a transformer).
    - **Optimal Value**: Typically set to match the model's input expectations. For transformers, values like 256, 512, or 768 are common. A default of 768 is suggested for consistency with `f_dim`.

5. **`g_dim (int)`**:
    - **Description**: Number of positional groups. This is useful when different positional encodings are grouped separately. For most applications involving 2D or 3D spatial data, `g_dim` is set to 1.
    - **Optimal Value**: Default is 1. It should be adjusted based on the specific architecture and requirements of the task.

6. **`gamma (float)`**:
    - **Description**: Variance scaling factor for initializing the Fourier feature weight matrix. This scales the frequencies and impacts how much the high-frequency components are emphasized.
    - **Optimal Value**: Common values range from 0.1 to 10. The paper suggests a default of 1.0, which provides a balanced initialization.

### Reflection on Values

- **`pos_dim`**: This value is task-specific. For most image processing tasks, `pos_dim = 2`.
- **`f_dim`**: Higher values provide more detailed embeddings but increase computational cost. A value of 768 is a good balance for detailed spatial encoding.
- **`h_dim`**: The hidden layer's size in the MLP affects the capacity to transform Fourier features. A default of 32 is efficient and effective.
- **`d_dim`**: This should match the expected input dimensionality for subsequent layers. A default of 768 aligns with common transformer models.
- **`g_dim`**: Typically 1 for most spatial tasks unless specific grouping is required.
- **`gamma`**: A value of 1.0 is balanced but can be tuned between 0.1 and 10 depending on the task's sensitivity to high-frequency components.
