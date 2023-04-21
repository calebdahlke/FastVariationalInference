import numpy as np
import plotly.graph_objects as go
import time
import pickle
import os
from datetime import datetime
import GMMMutualInfromationExperiment as util
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

###################################################################################
# This is the associated code for the following paper
# "Fast Variational Estimation of Mutual Information for Implicit and Explicit
# Likelihood Models" by Caleb Dahlke, Sue Zheng, and Jason Pacheco.
# This runs the extrapolation experiment in section 8.2
###################################################################################


def GenerateModel(d):
    ###############################################################################
    # Outline: Generate model parameters described in section 7.2
    #
    # Inputs:
    #       d - decision (control) point for an given MI
    #
    # Outputs:
    #       X_x - x relation to psi (in paper, x is called theta)
    #       X_d - y relation to psi for decision d
    #       Sigma_x - Covariance of x
    #       Sigma_d - Covariance of y for decision d
    ###############################################################################
    X_x = np.array([[1,-.5]])
    X_d = np.array([[-1, d]])
    Sigma_x = 3
    Sigma_d = 1#np.abs(np.sin(d))+1
    return X_x, X_d, Sigma_x, Sigma_d

def SampleJoint(mu_psi, Sigma_psi, X_x, X_d, Sigma_x, Sigma_d,N):
    ###############################################################################
    # Outline: Sample points from join by sampling from marginal psi and then 
    #          conditionals x|psi (theta|psi in paper) and y|psi,d
    #
    # Inputs:
    #       mu_psi - mean of marginal psi
    #       Sigma_psi - Covariance of marginal psi
    #       X_x - x relation to psi (in paper, x is called theta)
    #       X_d - y relation to psi for decision d
    #       Sigma_x - Covariance of x
    #       Sigma_d - Covariance of y for decision d
    #       N - number of samples
    #
    # Outputs:
    #       samples - (psi, x (theta), y)
    ###############################################################################
    Sample_psi = np.random.multivariate_normal(mu_psi.flatten(),Sigma_psi,N)
    Sample_x = np.random.normal((np.matmul(X_x,Sample_psi.T)[0].flatten())**2,np.sqrt(Sigma_x)).reshape((N,1))
    Sample_y = np.random.normal((np.matmul(X_d,Sample_psi.T)[0].flatten())**2,np.sqrt(Sigma_d)).reshape((N,1))
    samples = np.hstack((Sample_psi,Sample_x,Sample_y))
    return samples

def TrueMargEnt(samples,X_x):
    ###############################################################################
    # Outline: Uses sampled points to compute the nested Monte Carlo approach to
    #          approximating the marginal entropy via marginalizing out the
    #          nuisance variable, psi.
    #
    # Inputs:
    #       samples - sampled points for evaluation
    #       X_x - x relation to psi (in paper, x is called theta)
    #       Sigma_x - Covariance of x
    #
    # Outputs:
    #       MargEnt - NMC estimate of marginal entropy
    ###############################################################################
    N = len(samples)
    MargEnt = 0
    for i in range(N):
        p_xcondpsi = multivariate_normal.logpdf((np.matmul(X_x,samples[np.arange(N)!=i,0:2].T).flatten())**2,samples[i,2],Sigma_x)
        log_alpha = max(p_xcondpsi)
        lpx = logsumexp(p_xcondpsi - log_alpha) + log_alpha -np.log(N-1)
        MargEnt += (-1/N)*lpx
    return MargEnt

def TrueCondEnt(samples,X_x, X_d, Sigma_x, Sigma_d):
    ###############################################################################
    # Outline: Uses sampled points to compute the nested Monte Carlo approach to
    #          approximating the conditional entropy via marginalizing out the
    #          nuisance variable, psi.
    #
    # Inputs:
    #       samples - sampled points for evaluation
    #       X_x - x relation to psi (in paper, x is called theta)
    #       X_d - y relation to psi for decision d
    #       Sigma_x - Covariance of x
    #       Sigma_d - Covariance of y for decision d
    #
    # Outputs:
    #       CondEnt - NMC estimate of conditional entropy
    ###############################################################################
    N = len(samples)
    CondEnt = 0
    for i in range(N):
        p_xcondpsi = multivariate_normal.logpdf((np.matmul(X_x,samples[np.arange(N)!=i,0:2].T).flatten())**2,samples[i,2],Sigma_x)
        p_ycondpsi = multivariate_normal.logpdf((np.matmul(X_d,samples[np.arange(N)!=i,0:2].T).flatten())**2,samples[i,3],Sigma_d)
        
        # estimate joint (numerically stable)
        lintermed = p_xcondpsi + p_ycondpsi
        log_alpha = max(lintermed)        
        lpxy = logsumexp(lintermed - log_alpha) + log_alpha - np.log(N-1)
        
        # estimate marginal (numerically stable)
        log_beta = max(p_ycondpsi)
        lpy = logsumexp(p_ycondpsi - log_beta) + log_beta - np.log(N-1)
        
        #update entropy
        CondEnt += (-1/N)*(lpxy-lpy)
    return CondEnt

if __name__ == "__main__":

    # random.seed(10)
    PRINTRESULTS = False # Prints all the values of MI calculated
    PLOTEXAMPLE = True # Plots 4b and 4c (4a is always shown)

    ###################### Create GMM Model ######################################
    K = 15 # Number of Trials
    Ns = [200,400,1000,2000] # Number of Samples
    Tol = 10**(-6) # Tolerance for Gradient Method
    Dx = 1 # Dimension of x
    Dy = 1 # Dimension of y
    D = np.linspace( -3,3,13) # Decision points between -3 and 3
    mu_psi = np.array([[0],[0]]) # mean psi
    Sigma_psi = np.array([[2,1],[1,3]]) # covariance psi

    # Create storage for each method (Trials, Samples, Control Point, MI/Time)
    SampleMI = np.zeros((K,len(Ns),len(D),2))
    MMMI = np.zeros((K,len(Ns),len(D),2))
    GDMI = np.zeros((K,len(Ns),len(D),2))
    BAMI = np.zeros((K,len(Ns),len(D),2))
    VMMI = np.zeros((K,len(Ns),len(D),2))
    BAGD = np.zeros((K,len(Ns),len(D),2))
    VMGD = np.zeros((K,len(Ns),len(D),2))
    Mine = np.zeros((K,len(Ns),len(D),2))
    n=-1
    for N in Ns: # Loop over Number of Samples
        n+=1
        for k in range(K): # Loop over number or itterations
            for d in range(len(D)):
                X_x, X_d, Sigma_x, Sigma_d = GenerateModel(D[d])
                
                sampleStart = time.time()
                sample = SampleJoint(mu_psi, Sigma_psi,X_x, X_d, Sigma_x, Sigma_d,N)
                sampleEnd = time.time()
                sampleTime = sampleEnd-sampleStart
                if sampleTime<.00015:
                    sampleTime=.00015
                    
                ########################## Sample MI #########################################
                Stime0 = time.time()
                SampleMargEnt = TrueMargEnt(sample,X_x)
                SampleCondEnt = TrueCondEnt(sample,X_x, X_d, Sigma_x, Sigma_d)
                Stime1 = time.time()
                
                SampleMI[k,n,d,0] = SampleMargEnt-SampleCondEnt
                if (Stime1-Stime0)<.00015: # Lowest shown value on log plot
                    SampleMI[k,n,d,1] = .00015
                else:
                    SampleMI[k,n,d,1] = Stime1-Stime0+sampleTime
                ##############################################################################


                ########################## Implicit Likelihood MM #############################
                ILtime0 = time.time()
                EPX, EPY, EPXX, EPXY, EPYY = util.pmoments(sample[:,2:],Dx,Dy)

                ILMargEnt = util.MargEntMomentMatch(Dx, EPX, EPXX)
                ILCondEnt = util.CondEntMomentMatch(Dx, EPX, EPY, EPXX, EPXY, EPYY)
                ILtime1 = time.time()
                
                MMMI[k,n,d,0] = ILMargEnt-ILCondEnt
                MMMI[k,n,d,1] = ILtime1-ILtime0+sampleTime
                ##############################################################################


                ########################## Implicit Likelihood GD ############################
                GDtime0 = time.time()
                EPX, EPY, EPXX, EPXY, EPYY = util.pmoments(sample[:,2:],Dx,Dy)

                GDMargEnt = util.MargEntGradientDescent(Dx,EPX,EPXX,Tol)
                GDCondEnt = util.CondEntGradientDescent(Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY,Tol)
                GDtime1 = time.time()
                
                GDMI[k,n,d,0] = GDMargEnt-GDCondEnt
                GDMI[k,n,d,1] = GDtime1-GDtime0+sampleTime
                ##############################################################################
                

                ########################## Barber & Agakov MM #################################
                BAtime0 = time.time()
                EPX, EPY, EPXX, EPXY, EPYY = util.pmoments(sample[:,2:],Dx,Dy)

                BAMargEnt = TrueMargEnt(sample,X_x)
                BACondEnt = util.CondEntMomentMatch(Dx, EPX, EPY, EPXX, EPXY, EPYY)
                BAtime1 = time.time()
                
                BAMI[k,n,d,0] = BAMargEnt-BACondEnt
                BAMI[k,n,d,1] = BAtime1-BAtime0+sampleTime
                ##############################################################################


                ########################## Variational Marginal MM ###########################
                VMtime0 = time.time()
                EPX, EPY, EPXX, EPXY, EPYY = util.pmoments(sample[:,2:],Dx,Dy)

                VMMargEnt = util.MargEntMomentMatch(Dx, EPX, EPXX)
                VMCondEnt = TrueCondEnt(sample,X_x, X_d, Sigma_x, Sigma_d)
                VMtime1 = time.time()
                
                VMMI[k,n,d,0] = VMMargEnt-VMCondEnt
                VMMI[k,n,d,1] = VMtime1-VMtime0+sampleTime
                ##############################################################################
                
                ########################## Barber & Agakov GD ################################
                BAGDtime0 = time.time()
                EPX, EPY, EPXX, EPXY, EPYY = util.pmoments(sample[:,2:],Dx,Dy)

                BAGDMargEnt = TrueMargEnt(sample,X_x)
                BAGDCondEnt = util.CondEntGradientDescent(Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY,Tol)
                BAGDtime1 = time.time()
                
                BAGD[k,n,d,0] = BAGDMargEnt-BAGDCondEnt
                BAGD[k,n,d,1] = BAGDtime1-BAGDtime0+sampleTime
                ##############################################################################


                ####################### Variational Marginal GD ##############################
                VMGDtime0 = time.time()
                EPX, EPY, EPXX, EPXY, EPYY = util.pmoments(sample[:,2:],Dx,Dy)

                VMGDMargEnt = util.MargEntGradientDescent(Dx,EPX,EPXX,Tol)
                VMGDCondEnt = TrueCondEnt(sample,X_x, X_d, Sigma_x, Sigma_d)
                VMGDtime1 = time.time()
                
                VMGD[k,n,d,0] = VMGDMargEnt-VMGDCondEnt
                VMGD[k,n,d,1] = VMGDtime1-VMGDtime0+sampleTime

                ####################### MINE ##############################               
                MineTime0 = time.time()
                samplehold = sample[:,2:]
                data = (samplehold[:,:Dx],samplehold[:,Dx:])
                model = util.FullyConnected(var1_dim=Dx, var2_dim=Dy, L=2, H=np.array([10,10]))#[256]
                mine_obj = util.MINE(model, data, lr=5 * 1e-3)

                mine_obj.train(n_epoch=5000, batch_size=N)
                Mine[k,n,d,0] =np.mean(mine_obj.train_lb[-100:])
                MineTime1 = time.time()
                Mine[k,n,d,1] = MineTime1-MineTime0+sampleTime
###############################################################################                
                
########################## Debugging Prints ###################################
    if PRINTRESULTS:
        print("Sample MI:       %s" %SampleMI[k,n,5,0])
        print("Moment Match MI: %s" %MMMI[k,n,5,0])
        print("BA MI:           %s" %BAMI[k,n,5,0])
############################################################################### 
with open('SampleMI.pkl', 'wb') as f:
    pickle.dump(SampleMI, f)
with open('MMMI.pkl', 'wb') as f:
    pickle.dump(MMMI, f)
with open('GDMI.pkl', 'wb') as f:
    pickle.dump(GDMI, f)
with open('BAMI.pkl', 'wb') as f:
    pickle.dump(BAMI, f)
with open('VMMI.pkl', 'wb') as f:
    pickle.dump(VMMI, f)
with open('BAGD.pkl', 'wb') as f:
    pickle.dump(BAGD, f)
with open('VMGD.pkl', 'wb') as f:
    pickle.dump(BAGD, f)
with open('Mine.pkl', 'wb') as f:
    pickle.dump(Mine, f)
    
# with open('SampleMI.pkl', 'rb') as f:
#     SampleMI= pickle.load(f)
# with open('MMMI.pkl', 'rb') as f:
#     MMMI= pickle.load(f)
# with open('GDMI.pkl', 'rb') as f:
#     GDMI= pickle.load(f)
# with open('BAMI.pkl', 'rb') as f:
#     BAMI= pickle.load(f)
# with open('VMMI.pkl', 'rb') as f:
#     VMMI= pickle.load(f)
# with open('BAGD.pkl', 'rb') as f:
#     BAGD= pickle.load(f)
# with open('VMGD.pkl', 'rb') as f:
#     BAGD= pickle.load(f)
# with open('Mine.pkl', 'rb') as f:
#     Mine= pickle.load(f)

###############################################################################
############################# Plot 4a in Paper ################################
i=0
for N in Ns:
    SampleMIMean, SampleMIVariance = util.MeanAndVariance(len(D),K,SampleMI[:,i,:,:])
    MMMIMean, MMMIVariance = util.MeanAndVariance(len(D),K,MMMI[:,i,:,:])
    GDMIMean, GDMIVariance = util.MeanAndVariance(len(D),K,GDMI[:,i,:,:])
    BAMIMean, BAMIVariance = util.MeanAndVariance(len(D),K,BAMI[:,i,:,:])
    VMMIMean, VMMIVariance = util.MeanAndVariance(len(D),K,VMMI[:,i,:,:])
    BAGDMean, BAGDVariance = util.MeanAndVariance(len(D),K,BAGD[:,i,:,:])
    VMGDMean, VMGDVariance = util.MeanAndVariance(len(D),K,VMGD[:,i,:,:])
    MINEMean, MINEVariance = util.MeanAndVariance(len(D),3,Mine[:,i,:,:])
    xs = D.tolist()
    fig2 = go.Figure([
    go.Scatter(
        x=xs,
        y=SampleMIMean[:,0],
        line=dict(color='rgb(0,100,80)', width=3),
        mode='lines',
        name='I<sub>NMC</sub>'
    ),
    go.Scatter(
        x=xs+xs[::-1],
        y=np.hstack(((SampleMIMean[:,0]+SampleMIVariance[:,0]),(SampleMIMean[:,0]-SampleMIVariance[:,0])[::-1])), 
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ),
    go.Scatter(
        x=xs,
        y=MMMIMean[:,0],
        line=dict(color='rgb(255,0,0)', width=3),
        mode='lines',
        name='I<sub>m+p</sub> MM'
    ),
    go.Scatter(
        x=xs+xs[::-1],
        y=np.hstack(((MMMIMean[:,0]+MMMIVariance[:,0]),(MMMIMean[:,0]-MMMIVariance[:,0])[::-1])), 
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ),
    go.Scatter(
        x=xs,
        y=GDMIMean[:,0],
        line=dict(color='rgb(255,0,0)', width=3, dash='dot'),
        mode='lines',
        name='I<sub>m+p</sub> GD'
    ),
    go.Scatter(
        x=xs+xs[::-1], 
        y=np.hstack(((GDMIMean[:,0]+GDMIVariance[:,0]),(GDMIMean[:,0]-GDMIVariance[:,0])[::-1])), 
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ),
        go.Scatter(
        x=xs,
        y=BAMIMean[:,0],
        line=dict(color='rgb(0,0,255)', width=3),
        mode='lines',
        name='I<sub>post + NMC</sub> MM'
    ),
    go.Scatter(
        x=xs+xs[::-1], 
        y=np.hstack(((BAMIMean[:,0]+BAMIVariance[:,0]),(BAMIMean[:,0]-BAMIVariance[:,0])[::-1])), 
        fill='toself',
        fillcolor='rgba(0,0,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ),
    go.Scatter(
        x=xs,
        y=BAGDMean[:,0],
        line=dict(color='rgb(0,0,255)', width=3, dash='dot'),
        mode='lines',
        name='I<sub>post + NMC</sub> GD'
    ),
    go.Scatter(
        x=xs+xs[::-1], 
        y=np.hstack(((BAGDMean[:,0]+BAGDVariance[:,0]),(BAGDMean[:,0]-BAGDVariance[:,0])[::-1])), 
        fillcolor='rgba(0,0,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ),
        go.Scatter(
        x=xs,
        y=VMMIMean[:,0],
        line=dict(color='rgb(180,0,255)', width=3),
        mode='lines',
        name='I<sub>marg + NMC</sub> MM'
    ),
    go.Scatter(
        x=xs+xs[::-1], 
        y=np.hstack(((VMMIMean[:,0]+VMMIVariance[:,0]),(VMMIMean[:,0]-VMMIVariance[:,0])[::-1])), 
        fill='toself',
        fillcolor='rgba(180,0,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ),
        go.Scatter(
        x=xs,
        y=VMGDMean[:,0],
        line=dict(color='rgb(180,0,255)', width=3, dash='dot'),
        mode='lines',
        name='I<sub>marg + NMC</sub> GD'
    ),
    go.Scatter(
        x=xs+xs[::-1], 
        y=np.hstack(((VMGDMean[:,0]+VMGDVariance[:,0]),(VMGDMean[:,0]-VMGDVariance[:,0])[::-1])), 
        fill='toself',
        fillcolor='rgba(180,0,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ),
    go.Scatter(
        x=xs,
        y=MINEMean[:,0],
        line=dict(color='rgb(0,0,0)', width=3),
        mode='lines',
        name='MINE'
    ),
    go.Scatter(
        x=xs+xs[::-1], 
        y=np.hstack(((MINEMean[:,0]+MINEVariance[:,0]),(MINEMean[:,0]-MINEVariance[:,0])[::-1])), 
        fill='toself',
        fillcolor='rgba(0,0,0,0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo="skip",
        showlegend=False
    )
    ])
    fig2.update_xaxes(title_text="Decision")
    fig2.update_yaxes(title_text="MI Approximation")
    fig2.update_layout(font=dict(size=22),legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig2.update_layout(plot_bgcolor='white')
    fig2.update_layout(legend=dict(itemsizing='constant'))
    fig2.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig2.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    # fig2.write_image("ExtrapolationMIupdated.pdf")
    fig2.show()
    i+=1


########################## Plot Result ########################################
    if PLOTEXAMPLE:    
        SampleMIMean, SampleMIVariance = util.MeanAndVariance(len(Ns),K,SampleMI[:,:,7,:])
        MMMIMean, MMMIVariance = util.MeanAndVariance(len(Ns),K,MMMI[:,:,7,:])
        GDMIMean, GDMIVariance = util.MeanAndVariance(len(Ns),K,GDMI[:,:,7,:])
        BAMIMean, BAMIVariance = util.MeanAndVariance(len(Ns),K,BAMI[:,:,7,:])
        VMMIMean, VMMIVariance = util.MeanAndVariance(len(Ns),K,VMMI[:,:,7,:])
        BAGDMean, BAGDVariance = util.MeanAndVariance(len(Ns),K,BAGD[:,:,7,:])
        VMGDMean, VMGDVariance = util.MeanAndVariance(len(Ns),K,VMGD[:,:,7,:])
        MINEMean, MINEVariance = util.MeanAndVariance(len(Ns),3,Mine[:,:,7,:])
        
###############################################################################        
######################### Plot 4b in paper ####################################
        fig = go.Figure([
        go.Scatter(
            x=Ns,
            y=SampleMIMean[:,0],
            line=dict(color='rgb(0,100,80)', width=3),
            mode='lines',
            name='NMC'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], 
            y=np.hstack(((SampleMIMean[:,0]+SampleMIVariance[:,0]),(SampleMIMean[:,0]-SampleMIVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
        go.Scatter(
            x=Ns,
            y=MMMIMean[:,0],
            line=dict(color='rgb(255,0,0)', width=3),
            mode='lines',
            name='MM'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], 
            y=np.hstack(((MMMIMean[:,0]+MMMIVariance[:,0]),(MMMIMean[:,0]-MMMIVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
        go.Scatter(
            x=Ns,
            y=GDMIMean[:,0],
            line=dict(color='rgb(255,0,0)', width=3, dash='dot'),
            mode='lines',
            name='GD'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], 
            y=np.hstack(((GDMIMean[:,0]+GDMIVariance[:,0]),(GDMIMean[:,0]-GDMIVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
            go.Scatter(
            x=Ns,
            y=BAMIMean[:,0],
            line=dict(color='rgb(0,0,255)', width=3),
            mode='lines',
            name='BA+NMC'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], 
            y=np.hstack(((BAMIMean[:,0]+BAMIVariance[:,0]),(BAMIMean[:,0]-BAMIVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
            go.Scatter(
            x=Ns,
            y=BAGDMean[:,0],
            line=dict(color='rgb(0,0,255)', width=3, dash='dot'),
            mode='lines',
            name='BA+NMC GD'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], 
            y=np.hstack(((BAGDMean[:,0]+BAGDVariance[:,0]),(BAGDMean[:,0]-BAGDVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
            go.Scatter(
            x=Ns,
            y=VMMIMean[:,0],
            line=dict(color='rgb(180,0,255)', width=3),
            mode='lines',
            name='VM+NMC'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], 
            y=np.hstack(((VMMIMean[:,0]+VMMIVariance[:,0]),(VMMIMean[:,0]-VMMIVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(180,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
            go.Scatter(
            x=Ns,
            y=VMGDMean[:,0],
            line=dict(color='rgb(180,0,255)', width=3, dash='dot'),
            mode='lines',
            name='VM+NMC GD'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], 
            y=np.hstack(((VMGDMean[:,0]+VMGDVariance[:,0]),(VMGDMean[:,0]-VMGDVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(180,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
        go.Scatter(
            x=Ns,
            y=MINEMean[:,0],
            line=dict(color='rgb(0,0,0)', width=3),
            mode='lines',
            name='MINE'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], 
            y=np.hstack(((MINEMean[:,0]+MINEVariance[:,0]),(MINEMean[:,0]-MINEVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )
        ])
        fig.update_xaxes(title_text="Number of Samples", type="log", dtick = 1)
        fig.update_yaxes(title_text="MI Approximation")
        fig.update_layout(plot_bgcolor='white')
        fig.update_xaxes(
            range=[2.3, 3.3],#2,3.7
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_layout(showlegend=False)
        fig.update_layout(font=dict(size=30))
        # fig.write_image("ExtrapolationConvergeupdated.pdf")
        fig.show()

        ###############################################################################
        ######################### Plot 4c in paper ####################################
        fig1 = go.Figure([
        go.Scatter(
            x=Ns,
            y=SampleMIMean[:,1],
            line=dict(color='rgb(0,100,80)', width=3),
            mode='lines',
            name='I<sub>NMC</sub>'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], 
            y=np.hstack(((SampleMIMean[:,1]+SampleMIVariance[:,1]),(SampleMIMean[:,1]-SampleMIVariance[:,1])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
        go.Scatter(
            x=Ns,
            y=MMMIMean[:,1],
            line=dict(color='rgb(255,0,0)', width=3),
            mode='lines',
            name='I<sub>m+p</sub> MM'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], 
            y=np.hstack(((MMMIMean[:,1]+MMMIVariance[:,1]),(MMMIMean[:,1]-MMMIVariance[:,1])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
        go.Scatter(
            x=Ns,
            y=GDMIMean[:,1],
            line=dict(color='rgb(255,0,0)', width=3, dash='dot'),
            mode='lines',
            name='I<sub>m+p</sub> GD'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], 
            y=np.hstack(((GDMIMean[:,1]+GDMIVariance[:,1]),(GDMIMean[:,1]-GDMIVariance[:,1])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
            go.Scatter(
            x=Ns,
            y=BAMIMean[:,1],
            line=dict(color='rgb(0,0,255)', width=3),
            mode='lines',
            name='I<sub>post</sub> + NMC MM'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], 
            y=np.hstack(((BAMIMean[:,1]+BAMIVariance[:,1]),(BAMIMean[:,1]-BAMIVariance[:,1])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
            go.Scatter(
            x=Ns,
            y=BAGDMean[:,1],
            line=dict(color='rgb(0,0,255)', width=3, dash='dot'),
            mode='lines',
            name='I<sub>post</sub> + NMC GD'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], 
            y=np.hstack(((BAGDMean[:,1]+BAGDVariance[:,1]),(BAGDMean[:,1]-BAGDVariance[:,1])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
            go.Scatter(
            x=Ns,
            y=VMMIMean[:,1],
            line=dict(color='rgb(180,0,255)', width=3),
            mode='lines',
            name='I<sub>marg</sub> + NMC MM'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], 
            y=np.hstack(((VMMIMean[:,1]+VMMIVariance[:,1]),(VMMIMean[:,1]-VMMIVariance[:,1])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(180,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
            go.Scatter(
            x=Ns,
            y=VMGDMean[:,1],
            line=dict(color='rgb(180,0,255)', width=3, dash='dot'),
            mode='lines',
            name='I<sub>marg</sub> + NMC GD'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], 
            y=np.hstack(((VMGDMean[:,1]+VMGDVariance[:,1]),(VMGDMean[:,1]-VMGDVariance[:,1])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(180,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
            go.Scatter(
            x=Ns,
            y=MINEMean[:,1],
            line=dict(color='rgb(0,0,0)', width=3),
            mode='lines',
            name='MINE'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], 
            y=np.hstack(((MINEMean[:,1]+MINEVariance[:,1]),(MINEMean[:,1]-MINEVariance[:,1])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )
        ])
        fig1.update_xaxes(title_text="Number of Samples", type="log", dtick = 1)
        fig1.update_yaxes(title_text="Run Time", type="log", dtick = 1)#
        fig1.update_layout(plot_bgcolor='white')
        fig1.update_xaxes(
            range=[2.3, 3.3],#2,3.7
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig1.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig1.update_layout(showlegend=False)
        fig1.update_layout(font=dict(size=30))
        # fig1.write_image("ExtrapolationTimeupdated.pdf")
        fig1.show()
###############################################################################