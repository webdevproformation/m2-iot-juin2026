<?php

/* $tab = [];

for($i = 0 ; $i < 10000000000000 ; $i ++)
{
    $tab[] = $i ;
}

var_dump($tab);  */

function generate()
{
    for($i = 0 ; $i < 10000000 ; $i ++)
    {
       yield $i ;
    }
}