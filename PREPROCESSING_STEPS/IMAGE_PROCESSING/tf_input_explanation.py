"""
TensorFlow Input Layer Issues and Solutions
==========================================

Explains the ValueError you're seeing and provides solutions.

The error: "Sequential model has already been configured to use input shape" 
occurs because:

1. tf.keras.Input creates a symbolic tensor that defines the model's input
2. Sequential models get "built" the first time they see data
3. Once built, they cannot accept different input shapes
4. Your preprocessing pipeline creates Input layers with fixed shapes

SOLUTIONS DEMONSTRATED BELOW
"""

import tensorflow as tf
import numpy as np

def demonstrate_the_problem():
    """Show exactly why your code fails"""
    print("🚨 DEMONSTRATING THE PROBLEM")
    print("=" * 40)
    
    # This mimics your current approach
    print("Creating Sequential model with Input layer...")
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(427, 640, 3)),  # Fixed shape like your code
        tf.keras.layers.Rescaling(1./255)
    ])
    
    print("First call with matching shape:")
    data1 = tf.random.uniform((1, 427, 640, 3))
    result1 = model(data1)
    print(f"✅ Success: {data1.shape} → {result1.shape}")
    
    print("\nSecond call with different shape:")
    try:
        data2 = tf.random.uniform((1, 300, 300, 3))  # Different shape
        result2 = model(data2)
        print(f"Result: {data2.shape} → {result2.shape}")
    except Exception as e:
        print(f"❌ ERROR: {e}")

def solution_1_no_input_layer():
    """Solution 1: Remove Input layer from Sequential model"""
    print("\n🔧 SOLUTION 1: No Input Layer in Sequential")
    print("=" * 50)
    
    # Create model without Input layer
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Resizing(180, 180)
    ], name='flexible_preprocessor')
    
    print("Testing with different image shapes:")
    
    # Test different shapes
    shapes = [(1, 427, 640, 3), (1, 300, 300, 3), (1, 224, 224, 3)]
    
    for shape in shapes:
        data = tf.random.uniform(shape)
        result = model(data)
        print(f"✅ {shape} → {result.shape}")

def solution_2_functional_api():
    """Solution 2: Use Functional API with flexible inputs"""
    print("\n🔧 SOLUTION 2: Functional API with Variable Input")
    print("=" * 52)
    
    def create_flexible_model():
        # Variable input shape (None, None) for height and width
        inputs = tf.keras.Input(shape=(None, None, 3), name='flexible_input')
        
        # Apply processing layers
        x = tf.keras.layers.Rescaling(1./255)(inputs)
        x = tf.keras.layers.Resizing(180, 180)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=x, name='flexible_functional_model')
    
    model = create_flexible_model()
    
    print("Testing with different image shapes:")
    shapes = [(1, 427, 640, 3), (1, 300, 300, 3), (1, 1024, 768, 3)]
    
    for shape in shapes:
        data = tf.random.uniform(shape)
        result = model(data)
        print(f"✅ {shape} → {result.shape}")

def solution_3_layer_by_layer():
    """Solution 3: Apply layers individually without model"""
    print("\n🔧 SOLUTION 3: Individual Layer Application")
    print("=" * 48)
    
    # Create layers
    rescale_layer = tf.keras.layers.Rescaling(1./255)
    resize_layer = tf.keras.layers.Resizing(180, 180)
    
    def process_image(image):
        """Process image through layers individually"""
        x = rescale_layer(image)
        x = resize_layer(x)
        return x
    
    print("Testing with different image shapes:")
    shapes = [(1, 427, 640, 3), (1, 300, 300, 3), (1, 1024, 768, 3)]
    
    for shape in shapes:
        data = tf.random.uniform(shape)
        result = process_image(data)
        print(f"✅ {shape} → {result.shape}")

def your_error_explanation():
    """Explain exactly what's happening in your code"""
    print("\n📚 YOUR ERROR EXPLAINED")
    print("=" * 30)
    
    print("""
YOUR CURRENT CODE ISSUE:
========================

1. create_input_layer() creates tf.keras.Input(shape=(height, width, 3))
2. This Input layer FIXES the model to expect exactly (height, width, 3) 
3. When you process a 427×640 image, model expects (427, 640, 3)
4. Later processing creates model expecting different size
5. TensorFlow says: "Model already configured for (427, 640, 3)"

THE FIX:
========
Don't use tf.keras.Input in preprocessing Sequential models!
Use one of the solutions above instead.

YOUR SPECIFIC CASE:
==================
Your `create_input_layer(input_size_width, input_size_height)` creates:
- tf.keras.Input(shape=(input_size_height, input_size_width, 3))

This LOCKS the model to that exact size. Different image sizes = ERROR.

RECOMMENDED FIX FOR YOUR CODE:
=============================
Remove the Input layer from your Sequential model.
Let TensorFlow infer input shapes dynamically.
    """)

def fix_your_code_demo():
    """Show how to fix your specific code"""
    print("\n🛠️  FIXING YOUR SPECIFIC CODE")
    print("=" * 35)
    
    # Your current problematic approach (commented out)
    print("❌ Your current approach (causes error):")
    print("""
    # This WILL cause the error you're seeing:
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(height, width, 3)),  # PROBLEM!
        tf.keras.layers.Rescaling(1./255)
    ])
    """)
    
    print("\n✅ Fixed approach:")
    
    # Fixed approach
    def create_flexible_preprocessing_pipeline():
        """This works with any image size"""
        return tf.keras.Sequential([
            # No Input layer here!
            tf.keras.layers.Resizing(180, 180),  # Resize to standard size
            tf.keras.layers.Rescaling(1./255),   # Normalize
        ], name='flexible_preprocessor')
    
    model = create_flexible_preprocessing_pipeline()
    
    # Test with your actual image dimensions
    print("Testing with your image dimensions:")
    
    # Simulate your loaded_image shapes
    test_images = [
        tf.random.uniform((427, 640, 3)),    # Your doggy image
        tf.random.uniform((300, 400, 3)),    # Different image
        tf.random.uniform((1024, 768, 3)),   # Large image
    ]
    
    for i, image in enumerate(test_images, 1):
        print(f"\nImage {i}: Original shape {image.shape}")
        
        # Add batch dimension (like your code does)
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)
        
        # Process
        processed = model(image)
        
        # Remove batch dimension
        if processed.shape[0] == 1:
            processed = tf.squeeze(processed, 0)
        
        print(f"Image {i}: Processed shape {processed.shape}")
        print(f"Image {i}: Value range [{tf.reduce_min(processed):.3f}, {tf.reduce_max(processed):.3f}]")

def main():
    """Run all demonstrations"""
    print("🔍 TENSORFLOW INPUT LAYER ERROR ANALYSIS")
    print("=" * 50)
    
    # Show the problem
    demonstrate_the_problem()
    
    # Show solutions
    solution_1_no_input_layer()
    solution_2_functional_api()
    solution_3_layer_by_layer()
    
    # Explain your error
    your_error_explanation()
    
    # Fix your specific code
    fix_your_code_demo()
    
    print("\n🎯 SUMMARY FOR YOUR CODE:")
    print("=" * 30)
    print("✅ Remove tf.keras.Input from your Sequential model")
    print("✅ Let TensorFlow handle input shape inference automatically") 
    print("✅ Use Resizing layer as first layer to standardize dimensions")
    print("✅ Your preprocessing will work with any input image size")

if __name__ == "__main__":
    main()