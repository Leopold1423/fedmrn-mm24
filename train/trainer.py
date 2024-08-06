import torch
from model.srn import srn_set_probability


def train(net, trainloader, valloader, config, device: str = "cpu"):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["l2"])
    net.train()

    if config["com_type"] == "fedsrn": # fedsrn
        current_step = 0
        total_steps = len(trainloader) * config["epochs"]
    
    for _ in range(config["epochs"]):
        for images, labels in trainloader:
            if config["com_type"] == "fedsrn":  # fedsrn
                current_step += 1
                srn_set_probability(net, current_step/total_steps)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

    val_loss, val_acc = test(net, valloader, None, device)
    results = {"val_loss": val_loss, "val_accuracy": val_acc}
    return results

def test(net, testloader, steps: int = None, device: str = "cpu"):
    if len(testloader) == 0: return 0.0, 0.0
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if steps is not None and batch_idx == steps:
                break
    loss /= (batch_idx+1)
    accuracy = correct / total
    return loss, accuracy

