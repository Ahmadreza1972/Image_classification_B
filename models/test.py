#class Test:
#    def test_model(self,model, test_loader, criterion, device='cpu'):
#    model.to(device)
#    model.eval()  # Set model to evaluation mode
#    correct = 0
#    total = 0
#    running_loss = 0.0
#
#    with torch.no_grad():  # No gradients needed for testing
#        for inputs, labels in tqdm(test_loader, desc="Testing"):
#            inputs, labels = inputs.to(device), labels.to(device)
#
#            outputs = model(inputs)
#            loss = criterion(outputs, labels)
#            running_loss += loss.item()
#
#            _, predicted = torch.max(outputs, 1)
#            total += labels.size(0)
#            correct += (predicted == labels).sum().item()
#
#    accuracy = 100 * correct / total
#    print(f"Test Loss: {running_loss/len(test_loader):.4f}")
#    print(f"Test Accuracy: {accuracy:.2f}%")