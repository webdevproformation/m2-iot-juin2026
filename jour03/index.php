<?php

/*  $tab = [];

for($i = 0 ; $i < 1_000_000_000 ; $i ++)
{
    $tab[] = $i ;
}

var_dump($tab); */  

function generate(int $max)
{
    for($i = 0 ; $i < $max ; $i ++)
    {
       yield $i ;
    }
}

foreach(generate(1_000_000_000)  as $nb)
{
    echo $nb . PHP_EOL;
} 