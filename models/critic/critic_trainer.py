from os.path import join as pjoin
import torch
import torch.nn.functional as F
from tqdm import tqdm

def trainer_func(net, train_loader_iter, logger, optimizer, scheduler, args):
    ''' train stage1 DiffRoot
    '''
    net.train()
    for nb_iter in tqdm(range(1, args.total_iter+1), position=0, leave=True):
        batch = next(train_loader_iter)
        m1, m2, xt1, xt2, m_length1, m_length2, t1, t2 = batch
        input = {}
        input['motion_better'] = xt1
        input['motion_worse'] = xt2

        assert torch.all( t1-t2<0 ), t1-t2

        critic = net(input)
        loss, acc = loss_func(critic)

        # print(t1[0].item(), t2[0].item(), critic[0,0].item(), critic[0,1].item(), f'acc={acc}')

        # for name, param in net.named_parameters():
        #         if 'clip' not in name:
        #             print(name, param.requires_grad, param.grad is None)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        msg = f'Train. Iter {nb_iter} '
        msg += f" loss. {loss:.6f}, acc. {acc}"
        
        if nb_iter % args.print_iter ==  0 :
            logger.info(msg)

        if nb_iter % args.save_iter == 0:
            torch.save(net.module.state_dict(), pjoin(args.out_dir, 'net_last.pth'))


def loss_func(critic):

    critic_diff = critic[:, 0] - critic[:, 1]
    acc = torch.mean((critic_diff > 0).clone().detach().float())
    
    target = torch.zeros(critic.shape[0], dtype=torch.long).to(critic.device)
    loss_list = F.cross_entropy(critic, target, reduction='none')
    loss1 = torch.mean(loss_list)

    probs = torch.sigmoid(critic_diff)
    loss2 = -torch.mean(torch.log(probs + 1e-10))
    print(loss1.item(), loss2.item())
    return loss1, acc



