import torch
import time
import copy
from tqdm.auto import tqdm
from datetime import datetime


"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create a learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Initialize ModelTrainer
trainer = ModelTrainer()

# Train the model
trained_model, results = trainer.train(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    loss_fn=criterion,
    epochs=num_epochs,
    scheduler=scheduler,
    patience=10  # Early stopping patience
)

"""

class ModelTrainer:
    EARLY_STOPPING_PATIENCE = 10

    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_step(self, model, dataloader, loss_fn, optimizer):
        model.train()
        train_loss, train_acc = 0, 0
        
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        
        train_loss /= len(dataloader)
        train_acc /= len(dataloader)
        return train_loss, train_acc

    def test_step(self, model, dataloader, loss_fn):
        model.eval()
        test_loss, test_acc = 0, 0
        
        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                
                test_pred_logits = model(X)
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()
                
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
                
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        return test_loss, test_acc

    def train(self, model, train_loader, test_loader, optimizer, loss_fn, epochs, scheduler=None, patience=None):
        patience = patience if patience is not None else self.EARLY_STOPPING_PATIENCE
        model.to(self.device)
        results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        best_loss = float('inf')
        best_test_accuracy = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        early_stopping_counter = 0
        start_time = time.time()
        
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = self.train_step(model, train_loader, loss_fn, optimizer)
            test_loss, test_acc = self.test_step(model, test_loader, loss_fn)
            
            if scheduler:
                scheduler.step(test_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
                  f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f} | lr: {current_lr:.6f}")
            
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            
            if test_acc > best_test_accuracy:
                best_test_accuracy = test_acc
            
            if test_loss < best_loss:
                best_loss = test_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f'Early stopping counter: {early_stopping_counter} out of {patience}')
                if early_stopping_counter >= patience:
                    print('Early stopping triggered.')
                    break
        
        model.load_state_dict(best_model_wts)
        time_elapsed = time.time() - start_time
        print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        return model, results