"""
A dictionary of models users can choose between when using the app.

"""

model_index = {
    # The original post, reproduced with 1/10th of the total training epochs
    "models/MiniGrid-MemoryS7FixedStart-v0/WorkingModel.pt": "MemoryDT",
    "models/MiniGrid-MemoryS7FixedStart-v0/MemoryGatedMLP.pt": "MemoryDTGatedMLP",
    "models/MiniGrid-Dynamic-Obstacles-8x8-v0/ReproduceOriginalPostShort.pt": "DynamicObstaclesDT_reproduction",
}
