

#class Train:
#    def train_model(self,model, train_loader,val_loader, criterion, optimizer, device='cpu'):
#    model.to(device)
#    tr_los=[]
#    tr_ac=[]
#    val_los=[]
#    val_ac=[]
#    for epoch in range(self.epoch):
#        model.train()
#        correct = 0
#        total = 0
#        running_loss = 0.0
#        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epoch}"):
#            inputs, labels = inputs.to(device), labels.to(device)
#            # Zero the parameter gradients
#            optimizer.zero_grad()
#            # Forward pass
#            outputs = model(inputs)
#            loss = criterion(outputs, labels)
#            # Backward pass and optimization
#            loss.backward()
#            optimizer.step()
#            running_loss += loss.item()
#            _, predicted = torch.max(outputs, 1)
#            total += labels.size(0)
#            correct += (predicted == labels).sum().item()
#        accuracy = 100 * correct / total
#        tr_los.append(running_loss/len(train_loader))
#        tr_ac.append(accuracy)
#        model.eval()  # Set model to evaluation mode
#        val_correct = 0
#        val_total = 0
#        val_running_loss = 0.0
#        with torch.no_grad():  # No gradients needed for validation
#            for inputs, labels in val_loader:
#                inputs, labels = inputs.to(device), labels.to(device)
#                outputs = model(inputs)
#                loss = criterion(outputs, labels)
#                val_running_loss += loss.item()
#                _, predicted = torch.max(outputs, 1)
#                val_total += labels.size(0)
#                val_correct += (predicted == labels).sum().item()
#        val_accuracy = 100 * val_correct / val_total
#        val_los.append(val_running_loss/len(val_loader))
#        val_ac.append(val_accuracy)
#        print(f"Tr_Loss: {running_loss/len(train_loader):.4f}, val_loss: {val_running_loss/len(val_loader):.4f}, Tr_acc: {accuracy}, val_ac: {val_accuracy}")
#    return tr_ac,val_ac,tr_los,val_los