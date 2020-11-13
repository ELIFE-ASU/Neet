# from Doug Moore 12.18.2018
#
# modified by Bryan Daniels

import numpy as np
import networkx as nx
import pyinform as pi

from itertools import chain,combinations
import copy

#from neet.interfaces import is_network, is_fixed_sized
from neet.network import Network
from neet.synchronous import Landscape

def attractors_brute_force(net, size=None, subgraph=None):
    if not isinstance(net, Network):
        raise TypeError("net must be a network or a networkx DiGraph")
    
    if size is None:
        size = net.size

    pin = [ n for n in range(size) if n not in subgraph ]
    ls = Landscape(net,pin=pin)
    return ls.attractors

def greatest_predecessors(dag, n):
    pred = list(dag.predecessors(n))
    N = len(pred)
    greatest = []
    for i in range(N):
        is_greatest = True
        for j in range(N):
            if i != j and nx.has_path(dag, pred[i], pred[j]):
                is_greatest = False
                break
        if is_greatest:
            greatest.append(pred[i])

    return greatest


# modified from https://www.kkhaydarov.com/greatest-common-divisor-python/
def gcd(a,b):
    
    if a == 0 and b == 0:
        return 0
 
    while b != 0:
        new_a = b
        new_b = a % b
 
        a = new_a
        b = new_b
 
    return a

# from https://www.w3resource.com/python-exercises/python-basic-exercise-32.php
def lcm(x, y):
   if x > y:
       z = x
   else:
       z = y

   while(True):
       if((z % x == 0) and (z % y == 0)):
           lcm = z
           break
       z += 1

   return lcm

def _merge(nodes_module_1, attractor_module_1, attractor_module_2, overlap):
    """
    Merge two attractors from different modules.
    
    attractor_module_1 and attractor_module_2 should each be decoded attractors
    (numpy arrays of decoded states).
    
    Returns a list of attractors as a numpy array.
    Each attractor has shape (#timesteps)x(net.size)
    """
    size = len(attractor_module_1[0])
    # check for easier cases
    # (below code also works in these cases, but it is slower)
    if len(attractor_module_1) == 1 and len(attractor_module_2) == 1:
        if np.all( attractor_module_1[0,overlap] == attractor_module_2[0,overlap] ):
            result = np.empty([1,1,size])
            result[0,0,:] = attractor_module_2[0,:]
            result[0,0,nodes_module_1] = attractor_module_1[0,nodes_module_1]
            return result
        else:
            return []
    else:
        # BCD 12.21.2018 deal with attractors of differing length
        len1 = len(attractor_module_1)
        len2 = len(attractor_module_2)
        len_joint = lcm(len1,len2)
        num_joint = gcd(len1,len2)
        #if len1 > 1 or len2 > 1:
        #    print "len1 =",len1,"len2 =",len2,"len_joint =",len_joint,"num_joint =",num_joint
        result = np.empty([num_joint,len_joint,size])
        matched_attractor_count = 0 # not all potential attractors will match on overlap
        #joint1 = np.tile(attractor_module_1,len_joint/len1)
        #joint2 = np.tile(attractor_module_2,len_joint/len2)
        # loop over relative phase
        for attractor_index in range(num_joint):
            match = False
            #joint2_rolled = np.roll(joint2,attractor_index,axis=0)
            # loop over timesteps within the resulting merged attractor
            for time_index in range(len_joint):
                #state_module_1 = joint1[time_index]
                #state_module_2 = joint2_rolled[time_index]
                state_module_1 = attractor_module_1[time_index%len1]
                state_module_2 = attractor_module_2[(time_index+attractor_index)%len2]
                # BCD this may be overly conservative.  we might be able to get
                # away with checking only one state in the joint attractor
                if np.all( state_module_1[overlap] == state_module_2[overlap] ):
                    match = True
                    result[matched_attractor_count,time_index,:] = state_module_2[:]
                    result[matched_attractor_count,time_index,nodes_module_1] = \
                        state_module_1[nodes_module_1]
            if match:
                matched_attractor_count += 1
        return result[:matched_attractor_count]


def direct_sum(modules, attrsList, cksList=None, dynamic=False):
    """
    Returns a list of decoded attractors (and optionally a corresponding list
    of control kernels with the same length).  Each attractor is a numpy array of
    decoded states.
    
    modules                 : List of lists of nodes in each module.
                              Length #modules.  Each module has a varying number
                              of node indices.
    attrsList               : List of lists of attractors, one for each module.  
                              Each attractor is a numpy array of decoded states.
                              Length #modules.  Each attractor list has a varying
                              number of attractors, and each attractor has a varying
                              length.
    cksList (None)          : (optional) List of lists of sets of control kernel nodes.
                              If given, the function will also return a list of sets
                              of control kernel nodes corresponding to each returned
                              attractor.
                              Length #modules.  Each list of sets of control kernel
                              nodes has length equal to the number of attractors for
                              the corresponding module.  Each set of control kernel
                              nodes has a varying length.
    dynamic (False)         : If False, compute 'static' control kernels.
                              If True, compute 'dynamic' control kernels
                              (dynamic control kernels not yet supported).
    """
    assert(len(modules)==len(attrsList))
    if cksList is None:
        return_cks = False
        cksList = [ [ set() for attr in attrs ] for attrs in attrsList ]
    else:
        return_cks = True
    
    if len(modules) == 0:
        result = []
        result_cks = []
    elif len(modules) == 1:
        result = attrsList[0]
        result_cks = cksList[0]
    else:
        result = []
        result_cks = []
        attractors_of_remaining,cks_of_remaining = \
            direct_sum(modules[1:], attrsList[1:], cksList=cksList[1:],dynamic=dynamic)
        assert( len(attrsList[0]) == len(cksList[0]) )
        for attractor,ck in zip(attrsList[0],cksList[0]):
            subresult = []
            subresult_cks = []
            for attractor_of_remaining,ck_of_remaining \
                in zip(attractors_of_remaining,cks_of_remaining):
                allcomponentnodes = set.union(*modules[1:])
                overlap = list(set.intersection(allcomponentnodes,modules[0]))
                merged = _merge(list(modules[0]), attractor, attractor_of_remaining, overlap)
                
                # () merge control kernels
                if dynamic:
                    raise Exception("Dynamic control kernels not yet supported")
                # 3.29.2019 BCD if multiple valid attractors vary only in their
                # attractor_index (relative phase), then these attractors
                # do not have a static control kernel (I think).
                # This corresponds to len(merged) > 1.
                if len(merged) > 1:
                    ck_merged = None
                else:
                    # BCD 2.16.2019 I'm not completely convinced of the following line
                    ck_merged = merge_cks(ck,ck_of_remaining)

                if len(merged) > 0:
                    subresult += list(merged)
                    subresult_cks += [ ck_merged for m in merged ]
            if len(subresult) > 0:
                result += subresult
                result_cks += subresult_cks
        if [] in result: # sanity check
            raise Exception
        
    if return_cks:
        assert(len(result)==len(result_cks)) # sanity check
        return result,result_cks
    else:
        return result

def merge_cks(ck1,ck2):
    """
    Return union of two sets of control kernels.
    
    If either set is None, return None.  This handles cases for which
    a module does not have a valid control kernel.  In this case there
    is also no valid control kernel for the larger network.
    """
    if (ck1 == None) or (ck2 == None):
        return None
    else:
        return set.union(ck1,ck2)


# 12.19.2018 BCD
def remove_duplicates(attractors):
    # not sure the best way to do this.  for now, just remove if they share the
    # same first state
    first_states = [ a[0] for a in attractors ]
    attractor_dict = dict(zip(first_states,attractors))
    return attractor_dict.values()

# 12.19.2018 BCD
def all_ancestors(dag,module_number):
    """
    Return all ancestor modules of a given module
    """
    ancestors = nx.shortest_path(dag.reverse(),module_number).keys()
    return ancestors

def leaves(dag):
    """
    Find modules with zero out-degree.
    """
    lst = []
    for n in dag.nodes:
        if dag.out_degree(n) == 0:
            lst.append(n)
    return lst

def network_modules(net):
    """
    (Not actually used in attractors function at the moment---convenient 
    to see whether we expect modularity to speed things up significantly.)
    """
    if not isinstance(net,Network):
        raise TypeError("net must be a network or a networkx DiGraph")
    
    g = net.to_networkx_graph()
    
    modules = list(nx.strongly_connected_components(g))
    return modules

# taken from https://docs.python.org/3/library/itertools.html#recipes
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def pinning_produces_desired_attractor(net,pin,dynamic_pin_states,desired_attractor):
    """
    Tests whether a specific pinning (generally dynamic) produces a given
    desired attractor.
    
    pin                 : List of nodes to be pinned
    dynamic_pin_states  : Decoded binary states to which nodes are pinned.  Should have
                          shape (# timesteps)x(# of pinned nodes)
    desired_attractor   : List of encoded states representing the desired attractor
    """
    
    # find attractors with subset pinned
    subset_ls = Landscape(net,pin=pin,dynamic_pin=dynamic_pin_states)
    
    if len(subset_ls.attractors) == 1:
        # is the single attractor we found the one we want?
        pinned_attractor = subset_ls.attractors[0]
        return attractors_equivalent(pinned_attractor,desired_attractor)
    else:
        return False

def _combine_external_and_internal_pin(external_pin,external_pin_states,
    internal_pin,internal_pin_state):
    """
    Combine a specific (dynamical) pinning of external input nodes with
    a specific (static) pinning of a subset of internal nodes.
    
    external_pin            : List or set of pinned nodes considered external
    external_pin_states     : Decoded binary states to which external nodes are
                              _dynamically_ pinned.
                              Should have shape (# timesteps)x(total # of nodes in net)
    internal_pin            : List or set of pinned nodes considered internal
    internal_pin_state      : Decoded binary state to which internal nodes are
                              _statically_ pinned.
                              Should have shape (total # of nodes in net)
    
    Returns:
        pin                 : sorted list of pinned nodes
        dynamic_pin_states  : list with shape (# timesteps)x(total # of pinned nodes)
    
    Used in module_control_kernel.
    """
    # We first pin the external nodes
    pin = copy.deepcopy(list(external_pin))
    dynamic_pin_states = copy.deepcopy(external_pin_states)
    
    # now pin internal nodes with "static" pinning
    for node in internal_pin:
        pin += [node]
        for dynamic_pin_state in dynamic_pin_states:
            dynamic_pin_state[node] = internal_pin_state[node]
    
    dynamic_pin_states = [ [ state[i] for i in sorted(pin) ] for state in dynamic_pin_states ]
    
    return sorted(pin),dynamic_pin_states
    

# POTENTIAL FOR SPEEDUP: it is possible to have redundant
#                        cases where the subset nodes have the same
#                        values in distinct attractors; we could
#                        potentially save the landscape results and
#                        reuse them if we find another case where
#                        the pinning is equivalent.
def module_control_kernel(net,module,input,attractors_given_input,
    require_inputs=True,dynamic=False,verbose=False):
    """
    Find control kernel for a given module with a given input.  The control 
    kernel is defined (here) with respect to each attractor.  It is 
    the smallest subset of nodes that must be pinned
    in order to controllably reach that attractor in the unpinned network.
    
    Note that the control kernel is not necessarily unique.
    
    module                  : List of node indices
    input                   : List of decoded states representing the input attractor
    attractors_given_input  : List of decoded attractors.  Attractors can have
                              varying lengths.
    require_inputs (True)   : If True, automatically include input nodes as control kernel
                              nodes. This can be faster than searching through all
                              possibilities, and input nodes will always be control kernel
                              nodes as currently defined.
    dynamic (False)         : If False, finds "static" control kernel, meaning we
                              restrict pinning of control kernel nodes to be
                              constant in time.
                              If True, finds "dynamic" control kernel, meaning
                              control kernel nodes can be dynamically pinned.
                              (dynamic control kernel is not currently supported)
                             
    Returns a list of sets of node indices, one for each attractor.
    
    See also:
        sampled_control_kernel, which includes options for an "iterative" approach
        and modifying the "phenotype"
    """
    if dynamic:
        raise Exception("Dynamic control kernel is not currently supported.")
    
    if len(attractors_given_input) == 1:
        return [set()]
    elif len(attractors_given_input) == 0:
        raise Exception

    encode = net.state_space().encode

    external_pin = [ n for n in range(net.size) if n not in module ]

    num_attractors = len(attractors_given_input)

    # we'll find a control kernel for each attractor
    ck_list = [ -1 for i in range(num_attractors) ]
    
    if require_inputs:
        # automatically include input nodes
        required_nodes = tuple([ i for i in tuple(input_nodes(net)) if i in module ])
    else:
        required_nodes = ()
    
    # set up generators of distinguishing node sets
    # (all distinguishing nodes should be in the current module by
    # definition, so only search over these as "possible_nodes")
    distinguishing_nodes_gen_list = distinguishing_nodes_from_attractors(
                                        attractors_given_input,
                                        possible_nodes=module,
                                        required_nodes=required_nodes)
    
    # loop over attractors
    for attractor_index,desired_attractor in enumerate(attractors_given_input):
        
        distinguishing_nodes_gen = distinguishing_nodes_gen_list[attractor_index]
        
        # only nodes that are constant over the attractor can be control nodes
        const_nodes = list( np.where(np.var(desired_attractor,axis=0)==0)[0] )
        potential_control_nodes = [ i for i in const_nodes if i in module ]
        
        # if the attractor is a cycle, check whether it is controllable
        # when pinning _all_ constant nodes
        # (this is relatively fast to check and prevents worthless exploration
        # when there is no control kernel)
        if len(desired_attractor) > 1:
            subset = potential_control_nodes
            # combine dynamic external input and static internal pinning
            # Pin the subset to the first value in the desired attractor.
            pin,dynamic_pin_states = _combine_external_and_internal_pin(
                                        external_pin,input,
                                        subset,desired_attractor[0])
            desired_attractor_encoded = [ encode(state) for state in desired_attractor ]
            cycle_controllable = pinning_produces_desired_attractor(net,
                                    pin,dynamic_pin_states,desired_attractor_encoded)
            if not cycle_controllable:
                potential_control_nodes = []
                ck_list[attractor_index] = None
        # end cycle control kernel check
        
        # loop over sets of distinguishing nodes of increasing size
        # until we have found a control kernel
        while(ck_list[attractor_index] == -1):
            subset = distinguishing_nodes_gen.__next__()
            if len(subset) > 0:
                # combine dynamic external input and static internal pinning
                # Pin the subset to the first value in the desired attractor.
                pin,dynamic_pin_states = _combine_external_and_internal_pin(
                                            external_pin,input,
                                            subset,desired_attractor[0])
                desired_attractor_encoded = [ encode(state) for state in desired_attractor ]
                control_success = pinning_produces_desired_attractor(net,
                                        pin,dynamic_pin_states,desired_attractor_encoded)
                if control_success:
                    ck_list[attractor_index] = subset
        # end loop over subsets
    # end loop over attractors

    return ck_list
    
def modularize(net):
    """
    Returns: modules, dag
        where `modules' is a list of length equal to the number of modules, with each
        element a list of node indices belonging to that module;
        and `dag' is a directed acyclic graph in networkx format reperesenting dependencies
        among modules.
    """
    g = net.to_networkx_graph()
    modules = list(nx.strongly_connected_components(g))
    dag = nx.condensation(g)
    return modules, dag

def attractors(net, verbose=False, retall=False, find_control_kernel=False,
    encoded=True):
    """
    encoded (True)              : If True, states in returned attractors are encoded
                                  as integers.
                                  If False, states are not encoded (binary vectors).
    """
    if not isinstance(net, Network):
        raise TypeError("net must be a network or a networkx DiGraph")
    
    size = net.size
    modules,dag = modularize(net)
    dag_list = list(nx.topological_sort(dag))

    # attractors and control_kernels are indexed by module number
    attractors = {}
    control_nodes = {}
    delta_control_nodes = {}
    basin_entropies = {}

    if verbose: print("Modules =",modules)

    decode,encode = net.state_space().decode, net.state_space().encode
    decode_attractors = \
        lambda atts: [ np.array([decode(state) for state in att]) for att in atts ]
    encode_attractors = \
        lambda atts: [ [ encode(state) for state in att ] for att in atts ]

    ancestors_dict = {}

    for module_number in dag_list:
        if verbose:
            print
            print("Module number",module_number)
        parents = greatest_predecessors(dag, module_number)
        
        ancestors_dict[module_number] = set.union( set.union(*[modules[m] for m in all_ancestors(dag,module_number)]), modules[module_number] )

        nodes = modules[module_number]
        # all nodes not in the module will be pinned
        pin = [ n for n in range(size) if n not in nodes ]
        if verbose:
            if hasattr(net,'names'):
                print("nodes in the module:",[net.names[n] for n in nodes])
            else:
                print("nodes in the module:",nodes)

        # first, find inputs from ancestors (if any).
        # we store these in 'inputs', a list of numpy arrays of decoded states that below
        # will be used to (dynamically) set the values of nodes not in the current
        # module.
        # Each element of inputs is a numpy array with shape (#timesteps)x(net.size)
        # (#timesteps may vary across inputs)
        if len(parents) == 0: # the module has no ancestors
            # dynamically pin all other nodes to zero
            inputs = [np.zeros([1,net.size])]
            input_cks = [set()]
        else: # the module does have ancestors
            # want list of each parent's inclusive ancestors
            parent_ancestors = [ set.union(*[modules[pm] for pm in all_ancestors(dag,p)]) for p in parents ]
            if verbose:
                print("parent_ancestors =",parent_ancestors)
            
            parent_modules = [ modules[p] for p in parents ]
            parent_attractors = [ attractors[p] for p in parents ]
            if find_control_kernel:
                parent_cks = [ control_nodes[p] for p in parents ]
            else:
                parent_cks = None
            if verbose:
                print("parent_modules =",parent_modules)
                print("parent_attractors =",parent_attractors)
            inputs = direct_sum(parent_ancestors, parent_attractors, cksList=parent_cks)
            if find_control_kernel:
                inputs,input_cks = inputs
        if verbose:
            print("inputs =",inputs)
            if find_control_kernel:
                print("input control nodes =",input_cks)

        # now find attractors given each input
        attractor_list = []
        control_nodes_list = []
        new_control_nodes_list = []
        basin_entropies_list = []
        # we will add found attractors to a growing list---at the end the
        # list will include all attractors of the module over all possible inputs
        for input_index,input in enumerate(inputs):
            # dynamically pin parent nodes to attractor values
            # (input has shape (#timepoints)x(net.size)
            dynamic_pin = input[:,pin]
            ls = Landscape(net,pin=pin,dynamic_pin=dynamic_pin)
            decoded_attractors = decode_attractors(ls.attractors)
            attractor_list.extend( decoded_attractors )
            if verbose:
                print("input_states =",input_states)
                print("pin =",pin)
                print("dynamic_pin =",dynamic_pin)

            if find_control_kernel:
                # find control nodes of this module given this input.
                # new_control_nodes has length equal to that of ls.attractors
                new_control_nodes = module_control_kernel(net,nodes,input,decoded_attractors)
                new_control_nodes_list.extend(new_control_nodes)
                all_control_nodes = [ merge_cks(input_cks[input_index],cks) \
                    for cks in new_control_nodes ] # <-- I'm not sure I need to do this
                control_nodes_list.extend(all_control_nodes)
                basin_entropies_list.extend([ ls.basin_entropy() for att in ls.attractors])
                if verbose:
                    print("attractors given input =",ls.attractors)
                    if hasattr(net,'names'):
                        print("new control nodes =",[ [ net.names[i] for i in ck] for ck in new_control_nodes])
                    else:
                        print("new control nodes =",new_control_nodes)

        attractors[module_number] = attractor_list #remove_duplicates(attractor_list)
        control_nodes[module_number] = control_nodes_list
        delta_control_nodes[module_number] = new_control_nodes_list
        basin_entropies[module_number] = basin_entropies_list

    leaf_module_indices = leaves(dag)
    leaf_ancestors = [ ancestors_dict[m] for m in leaf_module_indices ]
    leaf_attractors = [ attractors[m] for m in leaf_module_indices ]
    if find_control_kernel:
        leaf_control_nodes = [ control_nodes[m] for m in leaf_module_indices ]
    else:
        leaf_control_nodes = None
    a = direct_sum(leaf_ancestors, leaf_attractors, cksList=leaf_control_nodes)
    if find_control_kernel:
        a,control_kernel_list = a

    if encoded:
        a = encode_attractors(a)

    if retall or find_control_kernel:
        outdict = {'attractors':attractors,
                   'ancestors_dict':ancestors_dict,
                   'modules':modules,
                   }
        if find_control_kernel:
            outdict.update({'control_kernels':control_kernel_list,
                            'basin_entropies':basin_entropies,
                            'delta_control_nodes':delta_control_nodes})
        return a,outdict
    else:
        return a



# 5.21.2019 sampling stuff below ----------------------------------------------

# see also https://stackoverflow.com/questions/2150108/efficient-way-to-shift-a-list-in-python
def rotate(lst, n):
    return lst[n:] + lst[:n]

def random_state(n):
    return np.random.randint(0,2,n)

def set_pin_state(state,pin,pin_state):
    for pin_idx,pin_val in zip(pin,pin_state):
        state[pin_idx] = pin_val
    return state

def attractor_from_initial_state(net,state,pin=[],pin_state=[]):
    encode = net.state_space().encode
    decode = net.state_space().decode
    if len(pin) > 0:
        state = set_pin_state(state,pin,pin_state)
    state_list = [encode(state),]
    new_state = -1
    while new_state not in state_list[:-1]:
        decoded_new_state = net.update(decode(state_list[-1]))
        if len(pin) > 0:
            decoded_new_state = set_pin_state(decoded_new_state,pin,pin_state)
        new_state = encode(decoded_new_state)
        state_list.append(new_state)
    #print state_list
    att = state_list[state_list.index(new_state)+1:]
    # ********* DEBUG
    #assert(len(pin)==len(pin_state))
    #consistent = np.all([ decode(att[0])[i] == j for i,j in zip(pin,pin_state) ])
    #if not consistent:
    #    print("pin = {}".format(pin))
    #    print("pin_state = {}".format(pin_state))
    #    print("att = {}".format(att))
    #    print("decode(att[0]) = {}".format(decode(att[0])))
    #    assert(consistent)
    # ********** END DEBUG
    return att

def input_nodes(net):
    """
    Returns a list of node indices corresponding to input nodes
    (those that have in-degree 1 consisting of a single self-loop.)
    """
    return np.where([ len(net.neighbors_in(i))==1 and (i in net.neighbors_in(i)) for i in range(net.size) ])[0]

def phenotype_projection(attractor,hidden_indices,net):
    """
    Project encoded state
    """
    encode,decode = net.state_space().encode,net.state_space().decode
    # decode
    att_decoded = [ decode(state) for state in attractor ]
    # project
    att_proj_decoded = copy.deepcopy(att_decoded)
    for idx in hidden_indices:
        for state in att_proj_decoded:
            state[idx] = 0
    # encode
    att_proj = [ encode(state) for state in att_proj_decoded ]
    return att_proj

def _standardize_attractor(att):
    """
    Return the standardized form of an encoded attractor.
    
    The standardized form begins in the state with the smallest encoded value.
    """
    list_att = list(att)
    # rotate att to standard form
    att_ID = attractor_ID(list_att)
    att_ID_index = list_att.index(att_ID)
    rotated_att = rotate(list_att,att_ID_index)
    return rotated_att

def attractors_equivalent(att1,att2):
    """
    Determine whether two encoded attractors are equivalent.
    
    Cyclic attractors are considered equivalent if they contain the same states in the
    same order.  For instance, [2,3,1] is equivalent to [1,2,3] (but not [3,2,1]).
    """
    if len(att1) != len(att2):
        return False
    else:
        att1standard = _standardize_attractor(att1)
        att2standard = _standardize_attractor(att2)
        for i in range(len(att1)):
            if att1standard[i] != att2standard[i]:
                return False
        return True
        
def atts_and_cks_equivalent(atts1,cks1,atts2,cks2,ck_size_only=False):
    """
    Check whether two sets of attractors and corresponding control kernels
    are equivalent.
    
    (Attractor,control kernel) pairs can be listed in any order.
    
    ck_size_only (False)        : If True, only the size of the control kernel must
                                  match for each attractor.
                                  If False, the exact nodes of the control kernel must
                                  match for each attractor.
    """
    if (len(atts1) != len(cks1)) or (len(atts2) != len(cks2)):
        raise ValueError
    if len(atts1) != len(atts2):
        return False
    else:
        atts2IDs = [attractor_ID(att2) for att2 in atts2]
        for att1,ck1 in zip(atts1,cks1):
            att1ID = attractor_ID(att1)
            # check that attractor with same ID exists
            if att1ID not in atts2IDs:
                return False
            idx2 = atts2IDs.index(att1ID)
            # check that attractors are equivalent
            if not attractors_equivalent(att1,atts2[idx2]):
                return False
            if ck_size_only:
                # check that size of control kernels are equal
                if ((ck1 == None) and (cks2[idx2] != None)) or \
                   ((ck1 != None) and (cks2[idx2] == None)):
                    return False
                elif (ck1 == None) and (cks2[idx2] == None):
                    pass
                elif len(ck1) != len(cks2[idx2]):
                    return False
            else:
                # check that control kernels are equivalent
                if ck1 != cks2[idx2]:
                    return False
        return True

def sampled_attractors(net,numsamples=10000,seed=123,pin=[],pin_state=[],phenotype_list=None,
                       return_unprojected=False,return_counts=False,desired_attractor=None):
    """
    Return unique attractors found via sampling.  Attractor states are defined by "phenotype"
    nodes.  
    
    Values of hidden nodes in returned attractors are set to zero.  To get sampled 
    values of hidden nodes, use 'return_unprojected'.
    
    pin ([])                  : List of indices of pinned nodes
    pin_state ([])            : List of values for pinned nodes (same length as 'pin')
    phenotype_list (None)     : List of indices of visible nodes.  If None, default
                                to all nodes.
    return_unprojected (False): If True, also return dictionary mapping encoded projected
                                states to sets of all seen decoded unprojected states.
    return_counts (False)     : If True, also return list of counts giving the number of
                                times each attractor was seen.
    desired_attractor (None)  : If given an attractor (a set or list of encoded states),
                                instead of returning all attractors, the function will return
                                True if the network has a single attractor phenotype that 
                                matches the given attractor on phenotype nodes, and False 
                                otherwise (in which case the run time can be much faster than 
                                finding all attractors).
                                
    (Note: I'm not sure if this is all consistent when there are nontrivial dynamics 
    in hidden nodes---e.g. it's possible that fixed point phenotypes could result from 
    limit cycles in hidden states)
    """
    # 5.21.2019 TO DO: Encapsulate "rotate attractor to standard form"
    
    if len(pin) != len(pin_state):
        raise Exception
    if type(pin) != list or type(pin_state) != list:
        raise TypeError("pin and pin_state must be lists")
    
    # deal with phenotype
    if phenotype_list is None:
        phenotype_list = range(net.size)
    else:
        # check phenotype_list is consistent
        for idx in phenotype_list:
            if idx not in range(net.size):
                raise Exception("phenotype_list not consistent with given net")
    non_phenotype_list = [ i for i in range(net.size) if i not in phenotype_list ]
    
    # standardize desired_attractor if given
    if desired_attractor is not None:
        # project desired_attractor to given phenotype
        desired_attractor = phenotype_projection(desired_attractor,non_phenotype_list,net)
        # rotate desired_attractor to standard form
        att_ID = attractor_ID(list(desired_attractor))
        att_ID_index = desired_attractor.index(att_ID)
        desired_attractor = rotate(list(desired_attractor),att_ID_index)
    
    encode,decode = net.state_space().encode,net.state_space().decode
    np.random.seed(seed)
    all_attractors,all_attractors_IDs = [],[]
    if return_unprojected: unprojected_dict = {}
    for i in range(numsamples):
        att = attractor_from_initial_state(net,random_state(net.size),pin=pin,pin_state=pin_state)
        #all_attractors.append( att )
        
        # do phenotype projection: set all non-phenotype nodes to a constant (0)
        att_proj = phenotype_projection(att,non_phenotype_list,net)
        att_ID = attractor_ID(att_proj) # use minimum projected state as ID
        
        if desired_attractor is None:
            all_attractors_IDs.append( att_ID )
            all_attractors.append( att_proj )
        else: # just determine whether all sampled initial states go to the desired attractor
            # rotate attractor so that ID state is first
            att_ID_index = att_proj.index(att_ID)
            att_proj = rotate(att_proj,att_ID_index)
            if desired_attractor != att_proj:
                return False
        
        if return_unprojected:
            att_decoded = [ decode(state) for state in att ]
            # rotate attractor so that ID state is first
            att_ID_index = att_proj.index(att_ID)
            att_decoded = rotate(att_decoded,att_ID_index)
            # convert to tuple form
            tuple_att_decoded = tuple([ tuple(state) for state in att_decoded])
            
            # add attractor to dictionary mapping ID to rotated, decoded, unprojected states
            if att_ID in unprojected_dict:
                unprojected_dict[att_ID].add( tuple_att_decoded )
            else:
                unprojected_dict[att_ID] = set([ tuple_att_decoded ])
    
    if desired_attractor is not None:
        # if we've made it here, we've successfully tested all starting points
        return True
    unique_IDs,unique_attractor_indices,unique_counts = \
        np.unique(all_attractors_IDs,return_index=True,return_counts=True)
    unique_atts = [ all_attractors[idx] for idx in unique_attractor_indices ]

    if return_unprojected and return_counts:
        return unique_atts,unprojected_dict,unique_counts
    elif return_unprojected:
        return unique_atts,unprojected_dict
    elif return_counts:
        return unique_atts,unique_counts
    else:
        return unique_atts
        #unique_IDs = np.unique(all_attractors_IDs)
        #return unique_IDs

def constant_nodes(att,net=None):
    """
    Return indices of nodes that are constant over time in the given attractor.
    
    net (None)              : If paseed a network, states in the attractor will be
                              decoded using net.state_space().decode.
                              If None, attractor is assumed to be decoded.
    """
    if net is not None:
        decode = net.state_space().decode
        decoded_att = np.array([ decode(state) for state in att ])
    else:
        decoded_att = att
    return list( np.where(np.var(decoded_att,axis=0)==0)[0] )

def attractor_ID(att):
    """
    Takes a single encoded attractor.
    """
    return min(att)

# ****************************************************************************************************************
# NOTE: This function is very similar to modularity.module_control_kernel, and should eventually be merged with it
# ****************************************************************************************************************
def sampled_control_kernel(net,numsamples=10000,phenotype='all',dynamic=False,seed=123,
    verbose=False,phenotype_only=False,require_inputs=False,
    iterative=False,iterative_rounds_out=True,
    _iterative_pin=[],_iterative_desired_att=None,_iterative_rounds=[]):
    """
    phenotype ('all')            : A list of node indices defining the phenotype, or a string:
                                   'internal' -> all non-input nodes 
                                   'all'      -> all nodes
    phenotype_only (False)       : If True, restrict control kernel nodes to only those nodes
                                   defining the phenotype.  If False, allow any node to be in the
                                   control kernel.
    require_inputs (False)       : If True, always include input nodes as control kernel nodes.
                                   This can be faster than searching through all possibilities,
                                   and input nodes will always be control kernel nodes in the
                                   limit of large numsamples.
    iterative (False)            : If True, use iterative procedure that starts by pinning a
                                   minimal set of distinguishing nodes, then pinning more
                                   distinguishing nodes if additional attractors remain.
                                   This is not guaranteed to find a minimally-sized control
                                   set in general.  This mode returns both the list of
                                   controlling sets and a list of the number of iterative
                                   rounds required for each attractor.
                                   If False, use the default method that pins distinguishing sets
                                   of increasing size in the original network.
    iterative_rounds_out (True)  : If True and iterative=True, also return a list of
                                   list of the number of attractors remaining at each
                                   iterative step for each attractor.
    _iterative_pin ([])          : (Used internally when iterative = True)
    _iterative_desired_att (None): (Used internally when iterative = True)
    _iterative_rounds ([])       : (Used internally when iterative = True)
    """
    decode,encode = net.state_space().decode,net.state_space().encode
    
    # set phenotype as list of node indices
    if np.isreal(phenotype[0]):
        phenotype_nodes = phenotype
    elif phenotype == 'internal':
        phenotype_nodes = [ i for i in range(net.size) if i not in input_nodes(net) ]
    elif phenotype == 'all':
        phenotype_nodes = list(range(net.size))
    else:
        raise Exception("Unrecognized phenotype")
    
    if iterative:
        if phenotype_nodes != list(range(net.size)):
            raise Exception("iterative = True not yet supported with hidden nodes")
    
    # set up _iterative_pin_state
    _iterative_pin = sorted(_iterative_pin)
    if iterative and _iterative_desired_att is not None:
        _iterative_pin_state = [ decode(_iterative_desired_att[0])[j] for j in _iterative_pin ]
    else:
        _iterative_pin_state = []
    
    # find original attractors
    original_sampled_attractors,full_attractor_dict = sampled_attractors(net,
                           numsamples=numsamples,
                           seed=seed,phenotype_list=phenotype_nodes,
                           return_unprojected=True,
                           pin=_iterative_pin,pin_state=_iterative_pin_state)
    original_sampled_attractor_IDs = [ attractor_ID(att) for att in original_sampled_attractors ]
    if _iterative_desired_att is not None:
        if attractor_ID(_iterative_desired_att) not in original_sampled_attractor_IDs:
            print("sampled_control_kernel WARNING: _iterative_desired_att was not found in the pinned sampled attractors.  We will add it manually in order to continue with the calculation.  You may need to increase numsamples for consistent results.")
            original_sampled_attractors.append(_iterative_desired_att)
            full_attractor_dict[attractor_ID(_iterative_desired_att)] = [[decode(state) for state in _iterative_desired_att]]
    original_sampled_attractors_decoded = \
        [ [ decode(state) for state in att ] for att in original_sampled_attractors ]
    if verbose and not (iterative and _iterative_desired_att is not None):
        print("sampled_control_kernel: number of attractors = {}".format(
                                                    len(original_sampled_attractors)))
    
    # we'll find a control kernel for each attractor
    ck_list = [ -1 for i in range(len(original_sampled_attractors)) ]
    # we'll also keep track of how many iterative rounds we need in the iterative case
    iterative_rounds_list = [ -1 for i in range(len(original_sampled_attractors)) ]
    
    # define the set of potential control nodes
    if phenotype_only:
        possible_nodes = phenotype_nodes
    else:
        # we order potential nodes to check hidden ones [currently external] first
        # so they are preferred
        hidden_nodes =  [ i for i in range(net.size) if i not in phenotype_nodes ]
        possible_nodes = hidden_nodes + phenotype_nodes
        
    if require_inputs:
        # automatically include input nodes
        required_nodes = tuple(input_nodes(net))
    else:
        required_nodes = ()
        
    # set up generators of distinguishing node sets
    distinguishing_nodes_gen_list = distinguishing_nodes_from_attractors(
                                        original_sampled_attractors_decoded,
                                        possible_nodes=possible_nodes,
                                        required_nodes=required_nodes)
        
    # loop over phenotype attractors
    if iterative and _iterative_desired_att is not None:
        assert(attractor_ID(_iterative_desired_att) in original_sampled_attractor_IDs)
        attractor_IDs_to_analyze = [ attractor_ID(_iterative_desired_att) ]
    else:
        attractor_IDs_to_analyze = [ attractor_ID(att) for att in original_sampled_attractors ]
    for attractor_index in range(len(original_sampled_attractors)):
        maxSizeTested = -1
        
        desired_attractor = original_sampled_attractors[attractor_index]
        if attractor_ID(desired_attractor) in attractor_IDs_to_analyze:
            distinguishing_nodes_gen = distinguishing_nodes_gen_list[attractor_index]
            
            full_attractor_list = list( full_attractor_dict[attractor_ID(desired_attractor)] )
            
            # if the attractor is a cycle, check whether it is controllable
            # when pinning _all_ constant nodes (this is relatively fast to check
            # and prevents worthless exploration when there is no control kernel)
            if len(desired_attractor) > 1:
                full_attractor_index = 0
                cycle_controllable = False
                # only nodes that are constant over the attractor can be control nodes
                possible_control_nodes = \
                    [ i for i in constant_nodes(desired_attractor,net) if i in possible_nodes ]
                while(not cycle_controllable and full_attractor_index < len(full_attractor_list)):
                    # get full attractor that will be tried for pinning
                    desired_full_attractor = full_attractor_list[full_attractor_index]
                    subset = possible_control_nodes
                    
                    # for now there are no external nodes here
                    # (only nodes that we pinned already in previous iterations when iterative=True)
                    external_pin = _iterative_pin #[]
                    external_pin_states = [list(desired_full_attractor[0])]
                    pin,dynamic_pin_states = _combine_external_and_internal_pin(
                                                external_pin,external_pin_states,
                                                subset,desired_full_attractor[0])
                    # for now we only have static pinning
                    static_pin_state = dynamic_pin_states[0]
                    
                    cycle_controllable = sampled_attractors(net,numsamples=numsamples,
                                                        seed=seed,phenotype_list=phenotype_nodes,
                                                        pin=pin,pin_state=static_pin_state,
                                                        desired_attractor=desired_attractor)
                    full_attractor_index += 1
                if not cycle_controllable:
                    potential_control_nodes = []
                    ck_list[attractor_index] = None
                    iterative_rounds_list[attractor_index] = 0
            # end cycle control check
                    
            if verbose:
                print("    Searching for control of attractor ID",attractor_ID(desired_attractor))
                
            if iterative:
                subset_list,num_atts_pinned_list = [],[]
                
            # loop over sets of distinguishing nodes of increasing size
            while(ck_list[attractor_index] == -1):
                subset = distinguishing_nodes_gen.__next__()
                
                if len(subset) > maxSizeTested:
                    maxSizeTested = len(subset)
                    if iterative and len(subset_list) > 0:
                        # then we've checked all distinguishing node sets of minimal size
                        # and we haven't yet found a control kernel, so we need to iterate
                        best_dn_set_index = np.argmin(num_atts_pinned_list)
                        iterative_pin = set(_iterative_pin).union(subset_list[best_dn_set_index])
                        num_remaining = num_atts_pinned_list[best_dn_set_index]
                        if verbose:
                            print("    Minimum pinned number of attractors = {}.  Iterating after pinning {}...".format(num_remaining,iterative_pin))
                        pinned_ck,rounds = sampled_control_kernel(net,
                                          numsamples=numsamples,
                                          phenotype=phenotype,
                                          dynamic=dynamic,
                                          seed=seed,
                                          verbose=verbose,
                                          phenotype_only=phenotype_only,
                                          require_inputs=require_inputs,
                                          iterative=True,
                                          _iterative_pin=iterative_pin,
                                          _iterative_desired_att=desired_attractor,
                                          _iterative_rounds=_iterative_rounds+[num_remaining,])
                        assert(len(pinned_ck)==1)
                        assert(len(rounds)==1)
                        ck_list[attractor_index] = set.union(iterative_pin,pinned_ck[0])
                        iterative_rounds_list[attractor_index] = rounds[0]
                    elif verbose:
                        print("    Testing distinguishing node sets of size",maxSizeTested)
                if verbose and (ck_list[attractor_index] == -1):
                    print("        Trying pinning {}".format(subset))
                full_attractor_index = 0
                # have we found the control kernel yet for this attractor?
                # XXX BCD 8.9.2019 looking back at this code now---is there a possibility that one
                #                  of the settings of hidden nodes (corresponding to one of the
                #                  'desired_full_attractor's) has a smaller control kernel, but we
                #                  don't keep looking for it once we've found a control kernel for
                #                  one of the settings of hidden nodes?
                while(ck_list[attractor_index] == -1 and full_attractor_index < len(full_attractor_list)):
                    # get full attractor that will be tried for pinning
                    desired_full_attractor = full_attractor_list[full_attractor_index]
                    
                    # for now there are no external nodes here
                    # (only nodes that we pinned already in previous iterations when iterative=True)
                    external_pin = _iterative_pin #[]
                    external_pin_states = [list(desired_full_attractor[0])]
                    pin,dynamic_pin_states = _combine_external_and_internal_pin(
                                                external_pin,external_pin_states,
                                                subset,desired_full_attractor[0])
                    # for now we only have static pinning
                    static_pin_state = dynamic_pin_states[0]
                    
                    if iterative: # we need to count the number of attractors
                        atts_pinned = sampled_attractors(net,numsamples=numsamples,
                                                    seed=seed,phenotype_list=phenotype_nodes,
                                                    pin=pin,pin_state=static_pin_state)
                        num_atts_pinned = len(atts_pinned)
                        if num_atts_pinned == 1:
                            control_success = True
                        else:
                            control_success = False
                            # keep track of the number of attractors we find
                            # for each distinguishing node set of minimum size
                            subset_list.append(subset)
                            num_atts_pinned_list.append(num_atts_pinned)
                            
                    else: # 'guaranteed' method: we can simply stop if we find more than one attractor
                        control_success = sampled_attractors(net,numsamples=numsamples,
                                                    seed=seed,phenotype_list=phenotype_nodes,
                                                    pin=pin,pin_state=static_pin_state,
                                                    desired_attractor=desired_attractor)
                    if control_success:
                        if verbose:
                            print("        Successful control of attractor ID",
                                  attractor_ID(desired_attractor),"with nodes",subset)
                        ck_list[attractor_index] = set(subset)
                        iterative_rounds_list[attractor_index] = _iterative_rounds + [1,]
                    
                    # move to next full_attractor given this phenotype
                    full_attractor_index += 1
                # end loop over full_attractors
            # end loop over subsets
    # end loop over phenotype attractors

    # remove any remaining ck placeholders for attractors we didn't analyze
    # (in the 'iterative' option there are cases when we only analyze a single attractor)
    ck_list_reduced = [ ck for ck in ck_list if ck != -1 ]
    iterative_rounds_list_reduced = [ r for r in iterative_rounds_list if r != -1 ]
    assert(len(ck_list_reduced) == len(attractor_IDs_to_analyze))
    assert(len(iterative_rounds_list_reduced) == len(attractor_IDs_to_analyze))
    
    if iterative and iterative_rounds_out:
        return ck_list_reduced,iterative_rounds_list_reduced
    else:
        return ck_list_reduced


# 5.28.2019 branched from neet.synchronous.basin_entropy
def sampled_basin_entropy(net,numsamples=10000,seed=123,pin=[],pin_state=[],
    phenotype_list=None,base=2.0):
    """
    Estimate the basin entropy of the landscape [Krawitz2007]_.

    :param base: the base of the logarithm
    :type base: a number or ``None``
    :return: the estimated basin entropy of the landscape of type ``float``
    """
    _,basin_sizes = sampled_attractors(net,numsamples=numsamples,seed=seed,
                                       pin=pin,pin_state=pin_state,
                                       phenotype_list=phenotype_list,
                                       return_counts=True)
    dist = pi.Dist(basin_sizes)
    return pi.shannon.entropy(dist, b=base)


# 8.9.2019
def sampled_distinguishing_nodes(net,numsamples=10000,phenotype='all',
                                 seed=123,verbose=False,**kwargs):
    decode,encode = net.state_space().decode,net.state_space().encode
    
    # set phenotype as list of node indices
    if np.isreal(phenotype[0]):
        phenotype_nodes = phenotype
    elif phenotype == 'internal':
        phenotype_nodes = [ i for i in range(net.size) if i not in input_nodes(net) ]
    elif phenotype == 'all':
        phenotype_nodes = list(range(net.size))
    else:
        raise Exception("Unrecognized phenotype")
    
    original_sampled_attractors = sampled_attractors(net,numsamples=numsamples,
                                                     seed=seed,phenotype_list=phenotype_nodes)
    original_sampled_attractors_decoded = \
        [ [ decode(state) for state in att ] for att in original_sampled_attractors ]
    if verbose:
        print("sampled_distinguishing_nodes: original attractors:",original_sampled_attractors)

    return distinguishing_nodes_from_attractors(original_sampled_attractors_decoded,
                                                **kwargs)

def distinguishing_nodes_from_attractors(attractors,possible_nodes=None,required_nodes=()):
    """
    Given a list of decoded attractors, return for each attractor a generator
    that produces sets of distinguishing nodes of increasing size.
    
    We require distinguishing nodes here to be constant.
    
    required_nodes (())             : A tuple of node indices that will always be
                                      included as distinguishing nodes.
    """
    
    dn_list = []
    
    attractors_mean = np.array( [ np.mean(att,axis=0) for att in attractors ] )
    
    for attractor_index,desired_attractor in enumerate(attractors):
        dn_gen = distinguishing_nodes_from_attractor(attractors,attractor_index,
                 possible_nodes,required_nodes=required_nodes,
                 attractors_mean=attractors_mean)
        dn_list.append(dn_gen)
    
    return dn_list
        
def distinguishing_nodes_from_attractor(attractors,attractor_index,possible_nodes=None,
    required_nodes=(),attractors_mean=None,sets_to_test_first=[]):
    """
    Given a list of decoded attractors and index of a particular attractor, return
    a distinguishing nodes generator for that attractor, which produces sets of
    distinguishing nodes of increasing size.
    
    We require distinguishing nodes here to be constant.
    
    required_nodes (())             : A tuple of node indices that will always be
                                      included as distinguishing nodes.
    attractors_mean (None)          : To increase speed when running for a large number
                                      of individual attractor_index values and the same
                                      attractors, optionally pass
                                      attractors_mean =
                                      np.array( [ np.mean(att,axis=0) for att in attractors ] )
    sets_to_test_first ([])         : If given, test the given sets _before_ looping over
                                      sets of increasing size.  Note this will not generally
                                      produce distinguishing node sets of minimal size.
                                      Any required_nodes are not forced to be included
                                      for these sets.
    """
    try:
        N = len(attractors[0][0])
    except TypeError:
        raise TypeError("Unrecognized form of attractors list")
    if type(required_nodes) is not tuple:
        raise TypeError("required_nodes must be a tuple")
    if possible_nodes is None:
        possible_nodes = range(N)
    
    # 8.9.2019 take mean over states in each attractor.  then values that are not 0 or 1
    # are not constant over the attractor
    if attractors_mean is None:
        attractors_mean = np.array( [ np.mean(att,axis=0) for att in attractors ] )
    
    desired_attractor = attractors[attractor_index]
    
    # find which visible nodes are constant for this attractor
    # (and thus potential distinguishing nodes)
    constant_possible_nodes = [ i for i in constant_nodes(desired_attractor) \
                                if i in possible_nodes ]
    desired_attractor_mean = np.mean(desired_attractor,axis=0)
    
    # don't include required nodes (these will automatically be included below)
    constant_possible_nodes = [ n for n in constant_possible_nodes if (n not in required_nodes) ]
    
    # potential future speedup here if we intelligently deal with nodes that
    # are constant over the attractors.  note that we still need to include these
    # as potential distinguishing nodes, but they will never constitute
    # distinguishing node sets by themselves.
    #constant_across_attractors = constant_nodes([ att[0] for att in attractors ])
    
    for subset in sets_to_test_first:
        subset_with_required = subset # + required_nodes
        # determine whether this set is a distinguishing set
        distinguished_atts_union = np.array([])
        for subset_node in subset_with_required:
            distinguished_atts = np.where(attractors_mean[:,subset_node] != \
                desired_attractor_mean[subset_node])[0]
            distinguished_atts_union = \
                np.union1d(distinguished_atts_union,distinguished_atts)
        if len(distinguished_atts_union) == len(attractors) - 1:
            yield set(subset_with_required)
    
    # loop over subsets of increasing size
    sets_to_test = powerset( constant_possible_nodes )
    for subset in sets_to_test:
        subset_with_required = subset + required_nodes
        # determine whether this set is a distinguishing set
        distinguished_atts_union = np.array([])
        for subset_node in subset_with_required:
            distinguished_atts = np.where(attractors_mean[:,subset_node] != \
                desired_attractor_mean[subset_node])[0]
            distinguished_atts_union = \
                np.union1d(distinguished_atts_union,distinguished_atts)
        if len(distinguished_atts_union) == len(attractors) - 1:
            yield set(subset_with_required)
    
